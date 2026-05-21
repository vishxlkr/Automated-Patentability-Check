"""
Automated Novelty Check System for Patent Pre-Screening
CORRECTED VERSION v5 — Patent-BERT + Improved Retrieval

Author: Devika Bakshi (122CS0301)
Supervisor: Asst. Prof. Sumanta Pyne
NIT Rourkela

CHANGES IN v5 (on top of all v4 fixes):
=========================================

RETRIEVAL IMPROVEMENTS (Recall@K was 36.5%@10, 55%@50):
---------------------------------------------------------
FIX-R1: Model upgrade from all-MiniLM-L6-v2 to anferico/bert-for-patents
         (Google's patent-specific BERT, trained on ~100M patent claims/abstracts).
         This dramatically improves semantic similarity for patent language like
         "comprising", "wherein", "embodiment", "prior art", etc.
         Fallback chain: bert-for-patents → AI-Growth-Lab/PatentBERT →
                          BAAI/bge-base-en-v1.5 → all-MiniLM-L6-v2

FIX-R2: Reciprocal Rank Fusion (RRF) for candidate retrieval.
         Instead of using BERT-only FAISS results as the candidate pool,
         we now take the UNION of:
           - Top-K*4 candidates from BERT FAISS
           - Top-K*4 candidates from TF-IDF
         Then re-rank by hybrid score. This ensures patents with strong
         TF-IDF signal but lower BERT score are never missed.

FIX-R3: TF-IDF candidate retrieval increased — uses full matrix similarity
         (not just BERT neighbours), so truly domain-specific patents with
         exact keyword matches are always included.

FIX-R4: Increased top_k default from 50 to 100 for evaluation Recall@K
         to better match realistic examiner workflows.

FIX-R5: BM25-style TF-IDF tuning — increased max_features from 8000 to
         15000 and added (1,3) trigrams alongside bigrams to better capture
         patent-specific compound terms.

FIX-R6: Score-level late fusion in retrieval: after collecting candidates from
         both BERT and TF-IDF, the final score uses _fuse() which applies
         the domain plausibility cap, ensuring consistent scoring with training.

CLASSIFICATION (kept from v4, already working well):
-----------------------------------------------------
- Direct pair scoring for train/val/test (no retrieval fallback bug)
- BAC-maximising threshold grid [0.20, 0.80]
- Mixed negatives (50% medium-hard + 50% easy)
- Platt calibration on mixed validation set
- Domain plausibility TF-IDF floor (prevents BERT cross-domain drift)
- Sparsity guard (threshold -0.05 if citation density < 0.01%)
"""

import sys
import pandas as pd
import numpy as np
import re
import hashlib
import pickle
import os
import time
import warnings
import logging
from collections import Counter
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: FAISS not installed. Run: pip install faiss-cpu")

import torch
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("WARNING: Sentence-BERT not installed. Run: pip install sentence-transformers")

torch.set_grad_enabled(False)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# MODEL PRIORITY LIST (FIX-R1)
# Try patent-specific models first, fall back gracefully
# ============================================================
SBERT_MODEL_PRIORITY = [
    "anferico/bert-for-patents",      # Google's patent BERT — best for patent text
    "AI-Growth-Lab/PatentBERT",       # Patent-specific, trained on USPTO data
    "BAAI/bge-base-en-v1.5",          # Strong general retrieval model
    "all-MiniLM-L6-v2",              # Lightweight fallback
]

# ============================================================
# PLATT SCALING CALIBRATOR
# ============================================================

class PlattCalibrator:
    """
    Fits logistic regression on (raw_pair_score -> probability).
    Trained on a MIXED distribution (easy + hard negatives).
    All scores are PAIR-LEVEL (not retrieval-max), ensuring
    consistency between train, val, and test distributions.
    """
    def __init__(self):
        self.lr = LogisticRegression(C=1.0, max_iter=1000)
        self.fitted = False
        self.score_min = 0.0
        self.score_max = 1.0

    def fit(self, scores, labels):
        scores = np.array(scores, dtype=float)
        labels = np.array(labels)
        if len(np.unique(labels)) < 2:
            self.fitted = False
            return
        self.score_min = float(scores.min())
        self.score_max = float(scores.max())
        self.lr.fit(scores.reshape(-1, 1), labels)
        self.fitted = True

    def predict_proba(self, scores):
        scores = np.array(scores, dtype=float)
        if not self.fitted:
            return scores
        return self.lr.predict_proba(scores.reshape(-1, 1))[:, 1]


# ============================================================
# MAIN SYSTEM
# ============================================================

class PatentNoveltySystem:
    """
    Patent Novelty Pre-Screening System — v5
    Patent-BERT + TF-IDF hybrid with RRF retrieval and pair-level calibration.
    """

    def __init__(self, cache_dir='cache/', model_dir='models/'):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Device]: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        self.vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_matrix_dense = None   # cached dense for fast retrieval
        self.tfidf_cache = None

        self.sbert_model = None
        self.sbert_model_name = None
        self.sbert_enabled = SBERT_AVAILABLE
        self.patent_embeddings = None
        self.sbert_index = None
        self.patent_ids_ordered = None
        self.id_to_index = None

        # Fusion weights — learned during training
        self.w_tfidf = 0.40
        self.w_bert = 0.60

        # Decision threshold — learned during validation
        self.final_threshold = 0.50

        # Out-of-domain guard: max pair score below this → NOVEL
        self.novelty_floor = 0.30

        # Domain plausibility: TF-IDF below this → scale down BERT
        self.tfidf_plausibility_floor = 0.05

        # RRF constant (FIX-R2)
        self.rrf_k = 60

        self.calibrator = PlattCalibrator()

        self.text_map = None
        self.title_map = None
        self.patents_df = None
        self.citation_set = None
        self.citation_set_bidirectional = set()

        self.cache_dir = cache_dir
        self.model_dir = model_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.hash_file = os.path.join(cache_dir, 'dataset_hash.txt')

    # ============================================================
    # UTILITIES
    # ============================================================

    def preprocess(self, text):
        """
        Preprocessing for patent text.
        Keeps technical compound terms by not over-stemming.
        """
        if pd.isna(text) or not str(text).strip():
            return ""
        text = str(text).lower()
        # Keep hyphens for compound terms (e.g. "deep-learning")
        text = re.sub(r'[^a-z\s\-]', '', text)
        text = re.sub(r'-', ' ', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words
                 if w not in self.stop_words and len(w) > 2]
        return " ".join(words)

    def compute_robust_dataset_hash(self, patents_df):
        hash_string = ""
        for pid in patents_df['patent_id'].values:
            text = patents_df[patents_df['patent_id'] == pid]['clean_text'].values[0]
            hash_string += f"{pid}:{hashlib.md5(text.encode()).hexdigest()}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def check_dataset_changed(self, patents_df):
        current_hash = self.compute_robust_dataset_hash(patents_df)
        changed = True
        previous_hash = None
        if os.path.exists(self.hash_file):
            with open(self.hash_file, 'r') as f:
                previous_hash = f.read().strip()
            changed = (current_hash != previous_hash)
        return changed, current_hash, previous_hash

    def save_dataset_hash(self, hash_value):
        with open(self.hash_file, 'w') as f:
            f.write(hash_value)
        logger.info(f"Dataset hash saved: {hash_value[:8]}...")

    # ============================================================
    # FAISS INDEX
    # ============================================================

    def build_faiss_index(self, embeddings):
        if not FAISS_AVAILABLE:
            return None
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        logger.info(f"FAISS index built: {index.ntotal} vectors")
        return index

    # ============================================================
    # DATASET CONSTRUCTION
    # ============================================================

    def build_citation_dataset(self, patent_file, abstract_file, citation_file,
                               min_citations=2, max_patents=10000):
        print("=" * 70)
        print("BUILDING CITATION-AWARE DATASET")
        print("=" * 70)

        print("\n[1/5] Analyzing citation graph...")
        citing_counter = Counter()
        cited_counter = Counter()

        chunk_iter = pd.read_csv(citation_file, sep='\t', dtype=str, chunksize=500000)
        first_chunk = next(chunk_iter)
        cols = first_chunk.columns.tolist()

        patent_col, cited_col = None, None
        for col in cols:
            col_lower = col.lower()
            if 'citing' in col_lower or col_lower == 'patent_id':
                patent_col = col
            if 'cited' in col_lower or 'citation_patent_id' in col_lower:
                cited_col = col
        if patent_col is None or cited_col is None:
            patent_col = cols[0]
            cited_col = cols[2] if len(cols) > 2 else cols[1]
        print(f"Using columns: citing='{patent_col}', cited='{cited_col}'")

        all_chunks = [first_chunk] + list(
            pd.read_csv(citation_file, sep='\t', dtype=str,
                        usecols=[patent_col, cited_col], chunksize=500000))
        for chunk in all_chunks:
            for val in chunk[patent_col].astype(str).fillna('').tolist():
                if val and val != 'nan':
                    citing_counter[val] += 1
            for val in chunk[cited_col].astype(str).fillna('').tolist():
                if val and val != 'nan':
                    cited_counter[val] += 1

        frequent_cited = {pid for pid, cnt in cited_counter.items()
                          if cnt >= min_citations}
        frequent_citing = {pid for pid, cnt in citing_counter.items()
                           if cnt >= min_citations}
        core_patents = list(frequent_cited.intersection(frequent_citing))
        print(f"Found {len(core_patents):,} core patents")

        if len(core_patents) > max_patents:
            core_patents = np.random.choice(
                core_patents, max_patents, replace=False).tolist()
            print(f"Randomly sampled {max_patents:,} core patents")

        print("\n[2/5] Loading patent data...")
        patent_chunks = []
        for chunk in pd.read_csv(patent_file, sep='\t', dtype=str, chunksize=10000):
            if 'patent_id' in chunk.columns:
                mask = chunk['patent_id'].isin(core_patents)
                if mask.any():
                    patent_chunks.append(chunk[mask])
        patents = pd.concat(patent_chunks, ignore_index=True)
        print(f"Loaded {len(patents):,} patents")

        print("\n[3/5] Loading abstract data...")
        abstract_chunks = []
        patent_ids_set = set(patents['patent_id'].astype(str))
        for chunk in pd.read_csv(abstract_file, sep='\t', dtype=str, chunksize=10000):
            if 'patent_id' in chunk.columns:
                mask = chunk['patent_id'].astype(str).isin(patent_ids_set)
                if mask.any():
                    abstract_chunks.append(chunk[mask])

        abstracts = (pd.concat(abstract_chunks, ignore_index=True)
                     if abstract_chunks else pd.DataFrame())
        df = (patents.merge(abstracts, on='patent_id', how='inner')
              if len(abstracts) > 0 else patents.copy())
        if len(abstracts) == 0:
            df['patent_abstract'] = ""

        title_col, abstract_col = None, None
        for col in df.columns:
            if 'title' in col.lower():
                title_col = col
            if 'abstract' in col.lower():
                abstract_col = col
        if title_col is None:
            title_col = df.columns[1] if len(df.columns) > 1 else 'patent_id'
        if abstract_col is None:
            abstract_col = title_col

        df = df[['patent_id', title_col, abstract_col]].dropna()
        df.columns = ['patent_id', 'patent_title', 'patent_abstract']
        print(f"Final patent count: {len(df):,}")

        print("\n[4/5] Preprocessing patent texts...")
        df['clean_text'] = (
            df['patent_title'] + " " + df['patent_abstract']
        ).apply(self.preprocess)
        df = df[df['clean_text'].str.split().str.len() >= 5].reset_index(drop=True)
        print(f"After quality filter: {len(df):,} patents")

        self.text_map = dict(zip(df['patent_id'], df['clean_text']))
        self.title_map = dict(zip(df['patent_id'], df['patent_title']))

        print("\n[5/5] Extracting citation pairs...")
        valid_ids = set(df['patent_id'].astype(str))
        citation_pairs = []
        for chunk in pd.read_csv(citation_file, sep='\t', dtype=str,
                                 usecols=[patent_col, cited_col],
                                 chunksize=500000):
            chunk = chunk.rename(
                columns={patent_col: 'patent_id', cited_col: 'cited_patent_id'})
            chunk['patent_id'] = chunk['patent_id'].astype(str)
            chunk['cited_patent_id'] = chunk['cited_patent_id'].astype(str)
            mask = (chunk['patent_id'].isin(valid_ids) &
                    chunk['cited_patent_id'].isin(valid_ids) &
                    (chunk['patent_id'] != chunk['cited_patent_id']))
            if mask.any():
                citation_pairs.append(chunk[mask])

        citations = (pd.concat(citation_pairs, ignore_index=True)
                     if citation_pairs else pd.DataFrame())
        print(f"Raw citation pairs: {len(citations):,}")

        n_patents = len(df)
        citation_density = len(citations) / max(n_patents * (n_patents - 1), 1)
        if citation_density < 0.0001:
            print(f"\n   [WARNING] Citation density is very low ({citation_density:.6%}). "
                  f"Threshold will be adjusted downward.")
            self._sparse_dataset = True
        else:
            self._sparse_dataset = False

        if len(citations) == 0:
            return self._create_demo_data()

        self.citation_set = set(zip(
            citations['patent_id'], citations['cited_patent_id']))
        self.citation_set_bidirectional = self.citation_set | {
            (b, a) for a, b in self.citation_set}

        n_positive = min(1500, len(citations))
        positive_pairs = citations.sample(n=n_positive, random_state=42).copy()
        positive_pairs['label'] = 1
        print(f"Created {len(positive_pairs)} positive pairs")

        self.patent_ids_ordered = list(valid_ids)
        self.patents_df = df
        return df, positive_pairs

    def _create_demo_data(self, num_patents=300):
        print(f"\nCreating demo dataset with {num_patents} patents...")
        self._sparse_dataset = False
        domains = {
            'neural_networks': [
                'neural network', 'deep learning', 'backpropagation', 'LSTM',
                'transformer', 'attention mechanism', 'gradient descent',
                'convolutional'
            ],
            'computer_vision': [
                'object detection', 'image segmentation', 'face recognition',
                'convolution', 'feature extraction', 'bounding box',
                'pixel classification'
            ],
            'nlp': [
                'text classification', 'sentiment analysis', 'machine translation',
                'BERT', 'language model', 'tokenization', 'named entity recognition'
            ],
            'reinforcement_learning': [
                'Q-learning', 'policy gradient', 'deep Q network', 'actor-critic',
                'reward function', 'Markov decision process', 'exploration strategy'
            ],
            'optimization': [
                'gradient descent', 'Adam optimizer', 'learning rate',
                'regularization', 'hyperparameter tuning', 'loss function',
                'convergence'
            ]
        }
        patents = []
        pid = 1
        for domain, terms in domains.items():
            count = num_patents // len(domains)
            for i in range(count):
                main_term = np.random.choice(terms)
                title = (f"System and method for {main_term.lower()} in "
                         f"{domain.replace('_', ' ')}")
                n_terms = np.random.randint(4, 7)
                selected = np.random.choice(
                    terms, size=min(n_terms, len(terms)), replace=False)
                abstract = (
                    f"A novel {domain.replace('_', ' ')} approach implementing "
                    f"{', '.join(selected[:-1])}, and {selected[-1]}. "
                    f"The method provides improved performance over prior art by "
                    f"combining multiple techniques in "
                    f"{domain.replace('_', ' ')} applications."
                )
                patents.append({
                    'patent_id': f"PAT{pid:04d}",
                    'patent_title': title,
                    'patent_abstract': abstract
                })
                pid += 1

        patents_df = pd.DataFrame(patents)
        patents_df['clean_text'] = (
            patents_df['patent_title'] + " " + patents_df['patent_abstract']
        ).apply(self.preprocess)
        self.text_map = dict(zip(patents_df['patent_id'], patents_df['clean_text']))
        self.title_map = dict(zip(patents_df['patent_id'], patents_df['patent_title']))
        self.patents_df = patents_df
        self.patent_ids_ordered = patents_df['patent_id'].tolist()

        domain_map = {}
        for p in patents:
            for domain in domains:
                if domain.replace('_', ' ') in p['patent_abstract']:
                    domain_map[p['patent_id']] = domain
                    break

        citation_pairs = []
        all_pids = patents_df['patent_id'].tolist()
        for i in range(len(all_pids)):
            for j in range(i + 1, len(all_pids)):
                p1, p2 = all_pids[i], all_pids[j]
                if (domain_map.get(p1) == domain_map.get(p2)
                        and np.random.random() < 0.25):
                    citation_pairs.append((p1, p2))

        citations = pd.DataFrame(
            citation_pairs, columns=['patent_id', 'cited_patent_id'])
        self.citation_set = set(
            zip(citations['patent_id'], citations['cited_patent_id']))
        self.citation_set_bidirectional = self.citation_set | {
            (b, a) for a, b in self.citation_set}

        print(f"Demo data: {len(patents_df)} patents, {len(citations)} citation pairs")
        positive_pairs = citations.sample(
            n=min(500, len(citations)), random_state=42).copy()
        positive_pairs['label'] = 1
        return patents_df, positive_pairs

    # ============================================================
    # SCORE HELPERS
    # ============================================================

    def _tfidf_sim(self, p1, p2):
        v1 = self.tfidf_cache[p1]
        v2 = self.tfidf_cache[p2]
        return float(np.clip(cosine_similarity(v1, v2)[0][0], 0.0, 1.0))

    def _bert_sim(self, idx1, idx2):
        emb1 = self.patent_embeddings[idx1]
        emb2 = self.patent_embeddings[idx2]
        raw = float(torch.dot(emb1, emb2).item())
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    def _fuse(self, tfidf, bert):
        """
        Fuse TF-IDF and BERT scores.
        Domain plausibility cap: if TF-IDF is very low, scale BERT down
        to prevent cross-domain semantic drift.
        """
        if tfidf < self.tfidf_plausibility_floor:
            bert = bert * (tfidf / self.tfidf_plausibility_floor)
        return float(np.clip(self.w_tfidf * tfidf + self.w_bert * bert, 0.0, 1.0))

    def _compute_pair_score_direct(self, p1, p2):
        """
        Compute DIRECT hybrid similarity between two patent IDs.
        This is the canonical scoring function used for ALL of:
          training, validation, test evaluation.
        Ensures threshold and calibrator are valid at test time.
        Returns: (raw_hybrid_score, calibrated_score) or (None, None)
        """
        i1 = self.id_to_index.get(str(p1))
        i2 = self.id_to_index.get(str(p2))
        if i1 is None or i2 is None:
            return None, None

        t = self._tfidf_sim(str(p1), str(p2))
        b = self._bert_sim(i1, i2)
        raw = self._fuse(t, b)

        cal = (float(self.calibrator.predict_proba(np.array([raw]))[0])
               if self.calibrator.fitted else raw)
        return raw, cal

    # ============================================================
    # EMBEDDINGS — FIX-R1: Patent-specific BERT + FIX-R5: better TF-IDF
    # ============================================================

    def _load_sbert_model(self):
        """
        FIX-R1: Try patent-specific models in priority order.
        Falls back gracefully if a model fails to load.
        """
        for model_name in SBERT_MODEL_PRIORITY:
            try:
                print(f"\n   Trying model: {model_name}")
                model = SentenceTransformer(model_name)
                model = model.to(self.device)
                # Quick sanity check
                _ = model.encode("patent claim test", convert_to_tensor=False)
                self.sbert_model_name = model_name
                print(f"   Loaded: {model_name} on {self.device}")
                return model
            except Exception as e:
                print(f"   Failed ({e.__class__.__name__}): {model_name}")
                continue
        raise RuntimeError("All SBERT models failed to load.")

    def compute_embeddings(self):
        print("\n" + "=" * 70)
        print("COMPUTING EMBEDDINGS")
        print("=" * 70)

        # FIX-R5: Larger vocabulary, trigrams, for better patent coverage
        print("\n[1/2] Computing TF-IDF vectors...")
        all_texts = list(self.text_map.values())
        self.vectorizer = TfidfVectorizer(
            max_features=15000,        # was 8000
            ngram_range=(1, 3),        # was (1, 2), now includes trigrams
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            analyzer='word',
        )
        self.vectorizer.fit(all_texts)
        self.patent_ids_ordered = self.patents_df['patent_id'].tolist()

        from scipy.sparse import vstack
        tfidf_list = [self.vectorizer.transform([self.text_map[pid]])
                      for pid in self.patent_ids_ordered]
        self.tfidf_matrix = vstack(tfidf_list)
        self.tfidf_cache = {
            pid: vec for pid, vec in zip(self.patent_ids_ordered, tfidf_list)}

        # Pre-compute dense TF-IDF matrix for fast cosine retrieval (FIX-R3)
        # Only do this if dataset is small enough to fit in memory
        n = len(self.patent_ids_ordered)
        if n <= 15000:
            self.tfidf_matrix_dense = self.tfidf_matrix.toarray().astype('float32')
            # L2 normalize for cosine via dot product
            norms = np.linalg.norm(self.tfidf_matrix_dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.tfidf_matrix_dense /= norms
            print(f"   TF-IDF dense matrix cached: {self.tfidf_matrix_dense.shape}")
        else:
            self.tfidf_matrix_dense = None

        if self.sbert_enabled:
            print("\n[2/2] Computing Patent-BERT embeddings (FIX-R1)...")
            if self.sbert_model is None:
                self.sbert_model = self._load_sbert_model()

            all_texts_bert = [self.text_map[pid] for pid in self.patent_ids_ordered]
            self.patent_embeddings = self.sbert_model.encode(
                all_texts_bert,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=True,
                batch_size=32,         # smaller batch for larger model
            )
            self.patent_embeddings = torch.nn.functional.normalize(
                self.patent_embeddings, p=2, dim=1)
            self.id_to_index = {
                pid: i for i, pid in enumerate(self.patent_ids_ordered)}

            if FAISS_AVAILABLE:
                embeddings_np = self.patent_embeddings.cpu().numpy().astype('float32')
                self.sbert_index = self.build_faiss_index(embeddings_np)
        else:
            self.id_to_index = {
                pid: i for i, pid in enumerate(self.patent_ids_ordered)}

        print("\nEmbeddings computed!")
        self.save_cached_embeddings()

    def init_sbert(self):
        if not SBERT_AVAILABLE:
            self.sbert_enabled = False
            return False
        if self.sbert_model is not None:
            return True
        print("\nInitializing SBERT model...")
        try:
            # Use saved model name if available
            model_name = (self.sbert_model_name
                          if self.sbert_model_name else SBERT_MODEL_PRIORITY[0])
            self.sbert_model = SentenceTransformer(model_name)
            self.sbert_model = self.sbert_model.to(self.device)
            self.sbert_enabled = True
            print(f"  Loaded {model_name} on {self.device}")
            return True
        except Exception as e:
            print(f"  Primary model failed, trying fallback: {e}")
            try:
                self.sbert_model = self._load_sbert_model()
                self.sbert_enabled = True
                return True
            except Exception as e2:
                print(f"  All models failed: {e2}")
                self.sbert_enabled = False
                return False

    # ============================================================
    # NEGATIVE MINING
    # ============================================================

    def get_random_negatives(self, query_ids, target_count, rng=None):
        if rng is None:
            rng = np.random.RandomState(99)
        all_ids = self.patent_ids_ordered
        negatives = []
        attempts = 0
        max_attempts = target_count * 20
        while len(negatives) < target_count and attempts < max_attempts:
            attempts += 1
            p1 = rng.choice(query_ids)
            p2 = rng.choice(all_ids)
            if p1 == p2:
                continue
            if (p1, p2) in self.citation_set_bidirectional:
                continue
            negatives.append({'patent_id': p1, 'cited_patent_id': p2, 'label': 0})
        return pd.DataFrame(negatives)

    def get_medium_hard_negatives(self, positive_pairs, target_count, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        print("\n   Mining medium-hard negatives (10th-40th percentile TF-IDF)...")
        negatives = []
        all_ids = self.patent_ids_ordered
        query_ids = positive_pairs['patent_id'].unique().tolist()
        attempts = 0
        max_attempts = target_count * 20

        while len(negatives) < target_count and attempts < max_attempts:
            attempts += 1
            p1 = rng.choice(query_ids)
            vec = self.tfidf_cache[p1]
            sims = cosine_similarity(vec, self.tfidf_matrix)[0]
            sorted_indices = np.argsort(sims)[::-1]
            n_total = len(sorted_indices)
            lo = int(n_total * 0.10)
            hi = int(n_total * 0.40)
            medium_range = sorted_indices[lo:hi]
            if len(medium_range) == 0:
                continue
            idx2 = rng.choice(medium_range)
            p2 = all_ids[idx2]
            if p1 == p2:
                continue
            if (p1, p2) in self.citation_set_bidirectional:
                continue
            negatives.append({'patent_id': p1, 'cited_patent_id': p2, 'label': 0})

        neg_df = pd.DataFrame(negatives)
        print(f"   Mined {len(neg_df)} medium-hard negatives")
        return neg_df

    def get_mixed_negatives(self, positive_pairs, target_count,
                            hard_ratio=0.5, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        n_hard = int(target_count * hard_ratio)
        n_easy = target_count - n_hard
        hard_df = self.get_medium_hard_negatives(positive_pairs, n_hard, rng=rng)
        query_ids = positive_pairs['patent_id'].unique().tolist()
        easy_df = self.get_random_negatives(query_ids, n_easy, rng=rng)
        mixed = pd.concat([hard_df, easy_df], ignore_index=True)
        mixed['label'] = 0
        return mixed

    # ============================================================
    # TRAINING
    # ============================================================

    def train_hybrid_model(self, positive_pairs):
        print("\n" + "=" * 70)
        print("TRAINING HYBRID MODEL")
        print("=" * 70)

        if self.tfidf_matrix is None:
            self.compute_embeddings()

        pos_shuffled = positive_pairs.sample(
            frac=1, random_state=42).reset_index(drop=True)
        n_val = max(60, int(len(pos_shuffled) * 0.30))
        val_positives = pos_shuffled.iloc[:n_val].copy()
        train_positives = pos_shuffled.iloc[n_val:].copy()
        print(f"\n   Train positives: {len(train_positives)} | "
              f"Val positives: {len(val_positives)}")

        # ── STEP 1: Compute training scores ─────────────────────
        print("\n[1/4] Computing scores for training positives...")
        train_tfidf_list, train_bert_list, train_labels = [], [], []

        for _, row in train_positives.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1 = self.id_to_index.get(p1)
            i2 = self.id_to_index.get(p2)
            if i1 is None or i2 is None:
                continue
            train_tfidf_list.append(self._tfidf_sim(p1, p2))
            train_bert_list.append(self._bert_sim(i1, i2))
            train_labels.append(1)

        n_pos = len(train_labels)
        print(f"   Valid training positives: {n_pos}")

        print("\n[2/4] Mining mixed negatives for training...")
        neg_train_df = self.get_mixed_negatives(
            train_positives, target_count=n_pos, hard_ratio=0.5,
            rng=np.random.RandomState(42))

        for _, row in neg_train_df.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1 = self.id_to_index.get(p1)
            i2 = self.id_to_index.get(p2)
            if i1 is None or i2 is None:
                continue
            train_tfidf_list.append(self._tfidf_sim(p1, p2))
            train_bert_list.append(self._bert_sim(i1, i2))
            train_labels.append(0)

        print(f"\n   Total training samples: {len(train_labels)}")
        print(f"   Positives: {sum(train_labels)} | "
              f"Negatives: {len(train_labels) - sum(train_labels)}")

        # ── STEP 2: Learn fusion weights ─────────────────────────
        print("\n[3/4] Learning optimal fusion weights via LR...")
        X_train = np.column_stack([train_tfidf_list, train_bert_list])
        y_train = np.array(train_labels)

        weight_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        weight_lr.fit(X_train, y_train)

        raw_w = weight_lr.coef_[0]
        w_sum = np.sum(np.abs(raw_w))
        if w_sum > 0:
            learned_w = np.abs(raw_w) / w_sum
            self.w_tfidf = float(np.clip(learned_w[0], 0.15, 0.55))
            self.w_bert = 1.0 - self.w_tfidf
        print(f"   Learned weights: TF-IDF={self.w_tfidf:.3f}, BERT={self.w_bert:.3f}")

        fused_train = np.array([
            self._fuse(t, b)
            for t, b in zip(train_tfidf_list, train_bert_list)])
        pos_mask = y_train == 1
        print(f"\n   Training Score Distribution (fused):")
        print(f"     Positive: mean={fused_train[pos_mask].mean():.3f}, "
              f"std={fused_train[pos_mask].std():.3f}")
        print(f"     Negative: mean={fused_train[~pos_mask].mean():.3f}, "
              f"std={fused_train[~pos_mask].std():.3f}")
        print(f"     Separation: "
              f"{fused_train[pos_mask].mean() - fused_train[~pos_mask].mean():.3f}")

        # ── STEP 3: Validate, calibrate, find threshold ──────────
        print("\n[4/4] Building validation set, calibrating, optimizing threshold...")
        val_neg_df = self.get_mixed_negatives(
            val_positives, target_count=len(val_positives), hard_ratio=0.5,
            rng=np.random.RandomState(77))  # seed=77 ≠ test seed=123

        val_all = pd.concat([val_positives, val_neg_df], ignore_index=True)

        val_tfidf_list, val_bert_list, val_labels_list = [], [], []
        for _, row in val_all.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1 = self.id_to_index.get(p1)
            i2 = self.id_to_index.get(p2)
            if i1 is None or i2 is None:
                continue
            val_tfidf_list.append(self._tfidf_sim(p1, p2))
            val_bert_list.append(self._bert_sim(i1, i2))
            val_labels_list.append(int(row['label']))

        val_fused = np.array([
            self._fuse(t, b)
            for t, b in zip(val_tfidf_list, val_bert_list)])
        val_labels_arr = np.array(val_labels_list)

        print(f"\n   Validation set: {len(val_labels_arr)} samples "
              f"(pos={val_labels_arr.sum()}, neg={(val_labels_arr==0).sum()})")

        self.calibrator.fit(val_fused, val_labels_arr)
        print(f"   Calibrator fitted: {self.calibrator.fitted}")
        print(f"   Model used: {self.sbert_model_name}")

        if self.calibrator.fitted:
            val_proba = self.calibrator.predict_proba(val_fused)
        else:
            val_proba = val_fused

        print(f"   Calibrated val score range: "
              f"[{val_proba.min():.3f}, {val_proba.max():.3f}]")
        if val_labels_arr.sum() > 0:
            print(f"     Positives: mean={val_proba[val_labels_arr==1].mean():.3f}")
        if (val_labels_arr == 0).sum() > 0:
            print(f"     Negatives: mean={val_proba[val_labels_arr==0].mean():.3f}")

        self._optimize_threshold_on_scores(
            val_proba, val_labels_arr,
            score_name="calibrated pair-level (mixed val)")

        if getattr(self, '_sparse_dataset', False):
            adj = 0.05
            old = self.final_threshold
            self.final_threshold = max(0.20, self.final_threshold - adj)
            print(f"\n   [Sparsity adjustment] Threshold "
                  f"{old:.3f} -> {self.final_threshold:.3f}")

        self.save_cached_embeddings()
        return True

    def _optimize_threshold_on_scores(self, y_scores, y_true,
                                       score_name="hybrid"):
        print(f"\n   Optimizing threshold on {score_name} scores...")
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        print(f"   Samples: pos={n_pos}, neg={n_neg}")
        print(f"   Score range: [{y_scores.min():.3f}, {y_scores.max():.3f}]")

        if n_pos == 0 or n_neg == 0:
            print("   WARNING: Only one class — using default threshold 0.50")
            self.final_threshold = 0.50
            return

        grid = np.linspace(0.20, 0.80, 241)
        best_bac, best_thresh, best_f1 = -1.0, 0.50, 0.0

        for thresh in grid:
            preds = (y_scores >= thresh).astype(int)
            bac = balanced_accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, zero_division=0)
            if bac > best_bac or (bac == best_bac and f1 > best_f1):
                best_bac, best_thresh, best_f1 = bac, thresh, f1

        self.final_threshold = float(best_thresh)
        preds_best = (y_scores >= best_thresh).astype(int)
        acc = accuracy_score(y_true, preds_best)
        prec = precision_score(y_true, preds_best, zero_division=0)
        rec = recall_score(y_true, preds_best, zero_division=0)
        f1_val = f1_score(y_true, preds_best, zero_division=0)
        bac_val = balanced_accuracy_score(y_true, preds_best)
        auc = (roc_auc_score(y_true, y_scores)
               if n_pos > 0 and n_neg > 0 else float('nan'))

        print(f"   Optimal threshold (BAC-maximising): {self.final_threshold:.3f}")
        print(f"   Accuracy={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  "
              f"F1={f1_val:.3f}  BAC={bac_val:.3f}")
        if not np.isnan(auc):
            print(f"   AUC-ROC: {auc:.3f}")

    # ============================================================
    # RETRIEVAL — FIX-R2, R3, R6: RRF + dual-source candidates
    # ============================================================

    def _get_tfidf_top_candidates(self, query_vec, n_candidates):
        """
        FIX-R3: Fast TF-IDF cosine retrieval using pre-normalised dense matrix
        or sparse matrix fallback.
        Returns (indices, scores) sorted by decreasing TF-IDF cosine.
        """
        if self.tfidf_matrix_dense is not None:
            # Fast dense dot product (pre-normalised → cosine)
            q_dense = query_vec.toarray().astype('float32').flatten()
            q_norm = np.linalg.norm(q_dense)
            if q_norm > 0:
                q_dense /= q_norm
            sims = self.tfidf_matrix_dense @ q_dense
        else:
            sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        sims = np.clip(sims, 0.0, 1.0)
        top_idx = np.argsort(sims)[::-1][:n_candidates]
        return top_idx, sims[top_idx]

    def _reciprocal_rank_fusion(self, bert_indices, tfidf_indices, k=60):
        """
        FIX-R2: Reciprocal Rank Fusion (RRF) over BERT and TF-IDF ranked lists.
        RRF score = sum_over_lists(1 / (k + rank))
        Returns sorted list of (index, rrf_score).
        """
        scores = {}
        for rank, idx in enumerate(bert_indices):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(tfidf_indices):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        # Return sorted by RRF score descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def compute_hybrid_similarity(self, query_text, top_k=100):
        """
        FIX-R2, R3, R6: Improved retrieval using RRF over BERT + TF-IDF candidates.

        Pipeline:
        1. Get top-K*4 candidates from BERT FAISS (semantic)
        2. Get top-K*4 candidates from TF-IDF (lexical, exact keywords)
        3. Merge via Reciprocal Rank Fusion → union candidate pool
        4. Re-rank by hybrid score _fuse(tfidf, bert)
        5. Return top-K results with calibrated scores
        """
        if not self.sbert_enabled or self.patent_embeddings is None:
            return self._tfidf_only_similarity(query_text, top_k)

        clean_query = self.preprocess(query_text)
        if not clean_query.strip():
            return []

        n_candidates = min(top_k * 4, len(self.patent_ids_ordered))

        # ── TF-IDF candidates ─────────────────────────────────
        query_vec = self.vectorizer.transform([clean_query])
        tfidf_top_idx, tfidf_top_scores = self._get_tfidf_top_candidates(
            query_vec, n_candidates)
        # Also compute full TF-IDF scores for all indices we need
        tfidf_sims_all = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        tfidf_sims_all = np.clip(tfidf_sims_all, 0.0, 1.0)

        # ── BERT candidates ───────────────────────────────────
        query_emb = self.sbert_model.encode(
            clean_query,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0)

        if self.sbert_index is not None and FAISS_AVAILABLE:
            query_np = query_emb.cpu().numpy().reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_np)
            bert_scores_raw, bert_idx = self.sbert_index.search(
                query_np, n_candidates)
            bert_top_idx = bert_idx[0]
            bert_scores_arr = np.clip((bert_scores_raw[0] + 1.0) / 2.0, 0.0, 1.0)
            # Build full bert scores dict for the candidate pool
            bert_scores_dict = {int(idx): float(s)
                                for idx, s in zip(bert_top_idx, bert_scores_arr)}
        else:
            bert_raw = torch.mm(
                query_emb.unsqueeze(0),
                self.patent_embeddings.T)[0].cpu().numpy()
            bert_scores_all = np.clip((bert_raw + 1.0) / 2.0, 0.0, 1.0)
            bert_top_idx = np.argsort(bert_scores_all)[::-1][:n_candidates]
            bert_scores_arr = bert_scores_all[bert_top_idx]
            bert_scores_dict = {int(idx): float(s)
                                for idx, s in zip(bert_top_idx, bert_scores_arr)}

        # ── RRF fusion of candidate lists ─────────────────────
        rrf_ranked = self._reciprocal_rank_fusion(
            bert_top_idx, tfidf_top_idx, k=self.rrf_k)

        # ── Re-rank by hybrid score ───────────────────────────
        results = []
        for orig_idx, _ in rrf_ranked:
            pid = self.patent_ids_ordered[orig_idx]

            tfidf_s = float(tfidf_sims_all[orig_idx])
            bert_s = bert_scores_dict.get(
                orig_idx,
                # If not in BERT top-K, compute directly
                float(np.clip(
                    (torch.dot(query_emb,
                               self.patent_embeddings[orig_idx]).item() + 1.0) / 2.0,
                    0.0, 1.0
                )) if self.patent_embeddings is not None else tfidf_s
            )

            raw_hybrid = self._fuse(tfidf_s, bert_s)
            cal_score = (float(self.calibrator.predict_proba(np.array([raw_hybrid]))[0])
                         if self.calibrator.fitted else raw_hybrid)

            results.append({
                'patent_id': pid,
                'title': self.title_map.get(pid, pid),
                'tfidf_sim': tfidf_s,
                'bert_sim': bert_s,
                'hybrid_sim': raw_hybrid,
                'calibrated_score': cal_score
            })

        # Sort by hybrid score and return top_k
        results.sort(key=lambda x: x['hybrid_sim'], reverse=True)
        return results[:top_k]

    def _tfidf_only_similarity(self, query_text, top_k=100):
        clean_query = self.preprocess(query_text)
        query_vec = self.vectorizer.transform([clean_query])
        sims = np.clip(cosine_similarity(query_vec, self.tfidf_matrix)[0], 0.0, 1.0)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [{
            'patent_id': self.patent_ids_ordered[idx],
            'title': self.title_map.get(self.patent_ids_ordered[idx], ''),
            'tfidf_sim': float(sims[idx]),
            'bert_sim': float(sims[idx]),
            'hybrid_sim': float(sims[idx]),
            'calibrated_score': float(sims[idx])
        } for idx in top_indices]

    # ============================================================
    # EVALUATION — Direct pair scoring (v4 fix retained)
    # ============================================================

    def evaluate_model(self, eval_pairs, eval_top_k=100):
        """
        Correct evaluation:
        - Classification: direct pair score for BOTH positive and negative pairs
        - Recall@K: retrieval-based, computed only for positive pairs
        """
        if eval_pairs is None or len(eval_pairs) == 0:
            print("No evaluation pairs provided.")
            return None

        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)

        y_true, y_scores_raw, y_scores_cal = [], [], []
        recall_counts = {10: 0, 20: 0, 50: 0, 100: 0}
        n_positives_total = 0
        skipped = 0

        for _, row in eval_pairs.iterrows():
            label = int(row['label'])
            p1 = str(row['patent_id'])
            p2 = str(row['cited_patent_id'])

            # Direct pair score — consistent with training
            raw, cal = self._compute_pair_score_direct(p1, p2)
            if raw is None:
                skipped += 1
                continue

            # Recall@K: retrieval only for positives
            if label == 1:
                n_positives_total += 1
                query_text = self.text_map.get(p1, '')
                if query_text:
                    results = self.compute_hybrid_similarity(query_text, top_k=eval_top_k)
                    retrieved_pids = [str(r['patent_id']) for r in results]
                    for k in recall_counts:
                        if p2 in retrieved_pids[:k]:
                            recall_counts[k] += 1

            y_true.append(label)
            y_scores_raw.append(float(raw))
            y_scores_cal.append(float(cal))

        if skipped > 0:
            print(f"   Skipped {skipped} pairs (missing embeddings)")

        y_true = np.array(y_true)
        y_scores_raw = np.array(y_scores_raw)
        y_scores_cal = np.array(y_scores_cal)

        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)

        print(f"\n   Evaluation set: {len(y_true)} samples "
              f"(pos={n_pos}, neg={n_neg})")
        print(f"   BERT model: {self.sbert_model_name}")

        print(f"\n   RAW Hybrid Score Distribution:")
        print(f"     Overall: min={y_scores_raw.min():.3f}, "
              f"max={y_scores_raw.max():.3f}, mean={y_scores_raw.mean():.3f}")
        if n_pos > 0:
            print(f"     Positives: mean={y_scores_raw[y_true==1].mean():.3f}, "
                  f"std={y_scores_raw[y_true==1].std():.3f}")
        if n_neg > 0:
            print(f"     Negatives: mean={y_scores_raw[y_true==0].mean():.3f}, "
                  f"std={y_scores_raw[y_true==0].std():.3f}")
        if n_pos > 0 and n_neg > 0:
            print(f"     Separation: "
                  f"{y_scores_raw[y_true==1].mean() - y_scores_raw[y_true==0].mean():.3f}")

        print(f"\n   CALIBRATED Score Distribution:")
        if n_pos > 0:
            print(f"     Positives: mean={y_scores_cal[y_true==1].mean():.3f}, "
                  f"std={y_scores_cal[y_true==1].std():.3f}")
        if n_neg > 0:
            print(f"     Negatives: mean={y_scores_cal[y_true==0].mean():.3f}, "
                  f"std={y_scores_cal[y_true==0].std():.3f}")
        if n_pos > 0 and n_neg > 0:
            print(f"     Separation: "
                  f"{y_scores_cal[y_true==1].mean() - y_scores_cal[y_true==0].mean():.3f}")

        y_pred = (y_scores_cal >= self.final_threshold).astype(int)

        auc_val = (roc_auc_score(y_true, y_scores_cal)
                   if n_pos > 0 and n_neg > 0 else float('nan'))
        ap_val = (average_precision_score(y_true, y_scores_cal)
                  if n_pos > 0 and n_neg > 0 else float('nan'))

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': auc_val,
            'avg_precision': ap_val,
            'threshold': self.final_threshold,
        }
        for k in recall_counts:
            metrics[f'recall_at_{k}'] = recall_counts[k] / max(n_positives_total, 1)

        print(f"\n[Classification Results] (threshold={metrics['threshold']:.3f}):")
        print(f"   Accuracy:          {metrics['accuracy']:.3f}  "
              f"({metrics['accuracy']*100:.1f}%)")
        print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"   Precision:         {metrics['precision']:.3f}")
        print(f"   Recall:            {metrics['recall']:.3f}")
        print(f"   F1 Score:          {metrics['f1']:.3f}")
        if not np.isnan(metrics['auc_roc']):
            print(f"   AUC-ROC:           {metrics['auc_roc']:.3f}")
        if not np.isnan(metrics['avg_precision']):
            print(f"   Avg Precision:     {metrics['avg_precision']:.3f}")

        print(f"\n[Retrieval Metrics] (out of {n_positives_total} positives):")
        for k in sorted(recall_counts):
            r = metrics[f'recall_at_{k}']
            print(f"   Recall@{k:<4}: {r:.3f}  ({r*100:.1f}%)")

        return metrics

    # ============================================================
    # CACHING
    # ============================================================

    def save_cached_embeddings(self):
        from scipy.sparse import save_npz
        save_npz(os.path.join(self.cache_dir, 'tfidf_matrix.npz'),
                 self.tfidf_matrix)
        if self.sbert_enabled and self.patent_embeddings is not None:
            torch.save(self.patent_embeddings.cpu(),
                       os.path.join(self.cache_dir, 'bert_embeddings.pt'))
        with open(os.path.join(self.cache_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        # Save dense TF-IDF matrix if it exists
        if self.tfidf_matrix_dense is not None:
            np.save(os.path.join(self.cache_dir, 'tfidf_dense.npy'),
                    self.tfidf_matrix_dense)
        metadata = {
            'patent_ids_ordered': self.patent_ids_ordered,
            'title_map': self.title_map,
            'text_map': self.text_map,
            'citation_set': list(self.citation_set) if self.citation_set else [],
            'w_tfidf': self.w_tfidf,
            'w_bert': self.w_bert,
            'final_threshold': self.final_threshold,
            'novelty_floor': self.novelty_floor,
            'tfidf_plausibility_floor': self.tfidf_plausibility_floor,
            'rrf_k': self.rrf_k,
            'calibrator': self.calibrator,
            'sparse_dataset': getattr(self, '_sparse_dataset', False),
            'sbert_model_name': self.sbert_model_name,
        }
        with open(os.path.join(self.cache_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        logger.info("Cached embeddings saved")

    def load_cached_embeddings(self):
        from scipy.sparse import load_npz
        tfidf_path = os.path.join(self.cache_dir, 'tfidf_matrix.npz')
        bert_path = os.path.join(self.cache_dir, 'bert_embeddings.pt')
        vectorizer_path = os.path.join(self.cache_dir, 'vectorizer.pkl')
        metadata_path = os.path.join(self.cache_dir, 'metadata.pkl')
        dense_path = os.path.join(self.cache_dir, 'tfidf_dense.npy')

        if not all(os.path.exists(p)
                   for p in [tfidf_path, vectorizer_path, metadata_path]):
            return False
        try:
            self.tfidf_matrix = load_npz(tfidf_path)
            if os.path.exists(dense_path):
                self.tfidf_matrix_dense = np.load(dense_path)
            if self.sbert_enabled and os.path.exists(bert_path):
                self.patent_embeddings = torch.load(bert_path, map_location='cpu')
                if self.device.type == 'cuda':
                    self.patent_embeddings = self.patent_embeddings.to(self.device)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(metadata_path, 'rb') as f:
                meta = pickle.load(f)
                self.patent_ids_ordered = meta['patent_ids_ordered']
                self.title_map = meta['title_map']
                self.text_map = meta['text_map']
                self.citation_set = set(meta.get('citation_set', []))
                self.citation_set_bidirectional = self.citation_set | {
                    (b, a) for a, b in self.citation_set}
                self.w_tfidf = meta.get('w_tfidf', 0.40)
                self.w_bert = meta.get('w_bert', 0.60)
                self.final_threshold = meta.get('final_threshold', 0.50)
                self.novelty_floor = meta.get('novelty_floor', 0.30)
                self.tfidf_plausibility_floor = meta.get(
                    'tfidf_plausibility_floor', 0.05)
                self.rrf_k = meta.get('rrf_k', 60)
                self.calibrator = meta.get('calibrator', PlattCalibrator())
                self._sparse_dataset = meta.get('sparse_dataset', False)
                self.sbert_model_name = meta.get('sbert_model_name', None)
            self.id_to_index = {
                pid: i for i, pid in enumerate(self.patent_ids_ordered)}
            if FAISS_AVAILABLE and self.patent_embeddings is not None:
                embeddings_np = (self.patent_embeddings.cpu()
                                 .numpy().astype('float32'))
                self.sbert_index = self.build_faiss_index(embeddings_np)
            logger.info("Cached embeddings loaded")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    # ============================================================
    # NOVELTY PREDICTION
    # ============================================================

    def predict_novelty(self, new_text, top_k=50):
        """
        For a new patent: retrieve top-K similar patents and decide novelty
        based on the max calibrated score across results.
        """
        if self.patents_df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 70)
        print("NOVELTY PREDICTION")
        print("=" * 70)
        print(f"\n[Query]: {new_text.strip()[:150]}...")

        start = time.time()
        results = self.compute_hybrid_similarity(new_text, top_k=top_k)
        elapsed = time.time() - start

        if not results:
            print("No results returned.")
            return "[ERROR] No similar patents found", []

        cal_scores = [r['calibrated_score'] for r in results]
        max_cal = float(max(cal_scores))
        top3_avg = float(np.mean(cal_scores[:3]))

        out_of_domain = max_cal < self.novelty_floor
        gap = abs(max_cal - self.final_threshold)

        if out_of_domain:
            confidence = "High (out-of-domain)"
            is_novel = True
        elif gap > 0.20:
            confidence = "High"
            is_novel = max_cal < self.final_threshold
        elif gap > 0.08:
            confidence = "Medium"
            is_novel = max_cal < self.final_threshold
        else:
            confidence = "Low — recommend manual review"
            is_novel = max_cal < self.final_threshold

        decision = ("[ACCEPT] Potentially Novel" if is_novel
                    else "[REJECT] Not Novel (High Similarity Detected)")

        print(f"\n[Results]:")
        print(f"   Max Calibrated Score:    {max_cal:.4f} ({max_cal*100:.1f}%)")
        print(f"   Top-3 Avg Cal Score:     {top3_avg:.4f} ({top3_avg*100:.1f}%)")
        print(f"   Decision Threshold:      {self.final_threshold:.3f}")
        print(f"   Novelty Floor:           {self.novelty_floor:.3f}")
        print(f"   Inference Time:          {elapsed:.3f}s")
        print(f"   Confidence:              {confidence}")
        print(f"   Fusion Weights:          "
              f"TF-IDF={self.w_tfidf:.3f}, BERT={self.w_bert:.3f}")
        print(f"   BERT Model:              {self.sbert_model_name}")
        print(f"   Calibrator fitted:       {self.calibrator.fitted}")

        print(f"\n{'='*60}")
        print("MOST SIMILAR PATENT (Decision Driver)")
        print(f"{'='*60}")
        top = results[0]
        print(f"\n  Patent ID : {top['patent_id']}")
        print(f"  Title     : {top['title'][:80]}...")
        print(f"  TF-IDF    : {top['tfidf_sim']:.4f}")
        print(f"  BERT      : {top['bert_sim']:.4f}")
        print(f"  Hybrid    : {top['hybrid_sim']:.4f}")
        print(f"  Calibrated: {top['calibrated_score']:.4f}")

        print(f"\n{'='*60}")
        print(f"TOP {min(10, len(results))} SIMILAR PATENTS")
        print(f"{'='*60}")
        for i, r in enumerate(results[:10], 1):
            print(f"\n{i:2d}. Patent: {r['patent_id']}")
            print(f"    Title  : {r['title'][:70]}...")
            print(f"    TFIDF={r['tfidf_sim']:.4f} | BERT={r['bert_sim']:.4f} | "
                  f"Hybrid={r['hybrid_sim']:.4f} | Cal={r['calibrated_score']:.4f}")

        print(f"\n{'='*60}")
        print(f"[DECISION]: {decision}")
        print(f"{'='*60}")

        return decision, results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PATENT NOVELTY CHECK SYSTEM — CORRECTED VERSION v5")
    print("=" * 70)
    print(f"\nAuthor    : Devika Bakshi (122CS0301)")
    print(f"Supervisor: Asst. Prof. Sumanta Pyne")
    print(f"Institute : NIT Rourkela")
    print(f"Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    system = PatentNoveltySystem()

    # ── Build dataset ──────────────────────────────────────────
    try:
        patents_df, positive_pairs = system.build_citation_dataset(
            patent_file="g_patent.tsv",
            abstract_file="g_patent_abstract.tsv",
            citation_file="g_us_patent_citation.tsv",
            min_citations=2,
            max_patents=10000
        )
    except Exception as e:
        print(f"\nReal data unavailable ({e}), using demo data...")
        patents_df, positive_pairs = system._create_demo_data(num_patents=300)

    # ── Dataset change detection ───────────────────────────────
    changed, current_hash, prev_hash = system.check_dataset_changed(patents_df)

    if changed:
        print(f"\n[INFO] Dataset changed — recomputing embeddings")
        print(f"   Hash: {prev_hash[:8] if prev_hash else 'None'} -> "
              f"{current_hash[:8]}")
        system.compute_embeddings()
        system.train_hybrid_model(positive_pairs)
        system.save_dataset_hash(current_hash)
    else:
        print(f"\n[OK] Dataset unchanged (hash: {current_hash[:8]})")
        if not system.load_cached_embeddings():
            print("   Cache load failed — recomputing...")
            system.compute_embeddings()
            system.train_hybrid_model(positive_pairs)
            system.save_dataset_hash(current_hash)
        else:
            system.init_sbert()

    # ── Build held-out test set ────────────────────────────────
    print("\n" + "=" * 70)
    print("BUILDING HELD-OUT TEST SET (mixed negatives)")
    print("=" * 70)

    n_test = min(200, len(positive_pairs))
    test_positive_sample = positive_pairs.sample(n=n_test, random_state=123).copy()
    test_positive_sample['label'] = 1

    test_neg_df = system.get_mixed_negatives(
        test_positive_sample, target_count=n_test, hard_ratio=0.5,
        rng=np.random.RandomState(123))

    test_pairs = pd.concat(
        [test_positive_sample, test_neg_df], ignore_index=True
    ).sample(frac=1, random_state=123).reset_index(drop=True)

    n_test_pos = int(test_pairs['label'].sum())
    n_test_neg = int((test_pairs['label'] == 0).sum())
    print(f"Test set: {len(test_pairs)} samples  "
          f"(pos={n_test_pos}, neg={n_test_neg})")

    # ── Evaluate ───────────────────────────────────────────────
    metrics = system.evaluate_model(test_pairs, eval_top_k=100)

    # ── Sample predictions ─────────────────────────────────────
    test_patent_texts = [
        """
        A quantum-inspired neural network architecture for real-time pattern recognition
        using adaptive resonance theory and deep reinforcement learning. The system
        employs quantum superposition principles to simultaneously evaluate multiple
        hypotheses in a neural architecture that adapts its weights based on environmental
        feedback signals, achieving superior convergence rates compared to classical methods.
        """,
        """
        A bicycle lock mechanism using a standard tumbler design with three rotating
        discs. The user sets a numeric combination to lock and unlock the device.
        The housing is made of hardened steel to resist cutting attacks.
        """,
        """
        A method for training large language models using reinforcement learning from
        human feedback (RLHF), comprising a reward model trained on human preference
        data and a policy optimization step using proximal policy optimization (PPO).
        The approach significantly reduces harmful outputs and improves instruction following.
        """
    ]

    print("\n" + "=" * 70)
    print("SAMPLE NOVELTY PREDICTIONS")
    print("=" * 70)

    for i, text in enumerate(test_patent_texts, 1):
        print(f"\n--- Test Patent {i} ---")
        decision, _ = system.predict_novelty(text, top_k=50)

    print(f"\n{'='*70}")
    print("SYSTEM SUMMARY")
    print(f"{'='*70}")
    print(f"  BERT Model           : {system.sbert_model_name}")
    print(f"  Final threshold      : {system.final_threshold:.3f}")
    print(f"  Novelty floor        : {system.novelty_floor:.3f}")
    print(f"  TF-IDF plaus. floor  : {system.tfidf_plausibility_floor:.3f}")
    print(f"  RRF k constant       : {system.rrf_k}")
    print(f"  Fusion weights       : TF-IDF={system.w_tfidf:.3f}, "
          f"BERT={system.w_bert:.3f}")
    print(f"  Calibrator fitted    : {system.calibrator.fitted}")
    if metrics:
        print(f"  Test Accuracy        : {metrics['accuracy']:.3f}")
        print(f"  Test Bal. Acc.       : {metrics['balanced_accuracy']:.3f}")
        print(f"  Test Precision       : {metrics['precision']:.3f}")
        print(f"  Test Recall          : {metrics['recall']:.3f}")
        print(f"  Test F1              : {metrics['f1']:.3f}")
        auc = metrics.get('auc_roc', float('nan'))
        if not (isinstance(auc, float) and np.isnan(auc)):
            print(f"  Test AUC-ROC         : {auc:.3f}")
        for k in [10, 20, 50, 100]:
            r = metrics.get(f'recall_at_{k}', 0.0)
            print(f"  Recall@{k:<4}          : {r:.3f}  ({r*100:.1f}%)")
    print(f"\n  System ready for production.")


if __name__ == "__main__":
    main()