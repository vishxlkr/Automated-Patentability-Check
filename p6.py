"""
Automated Novelty Check System for Patent Pre-Screening
CORRECTED VERSION v9 — True Patent-BERT Integration + Dual-Mode Architecture

Author: Devika Bakshi (122CS0301)
Supervisor: Asst. Prof. Sumanta Pyne
NIT Rourkela

WHAT v9 ADDS OVER v8:
======================

1. DUAL-MODE ARCHITECTURE  (USE_PATENT_BERT flag)
   ─────────────────────────────────────────────────
   USE_PATENT_BERT = True   →  anferico/bert-for-patents  (domain-adapted)
   USE_PATENT_BERT = False  →  BAAI/bge-base-en-v1.5 + v8 drift safeguards

   The flag lives at the top of the file so the examiner / prof can flip it
   instantly to compare both modes side-by-side.

2. CORRECT EMBEDDING NORMALISATION FOR PATENT-BERT
   ─────────────────────────────────────────────────
   anferico/bert-for-patents is a raw BERT encoder, NOT an SBERT model.
   It does NOT apply mean-pooling + L2 normalisation by default.
   If you skip normalisation:
     • cosine similarity scores become unstable (range shifts unpredictably)
     • thresholds learned on un-normalised embeddings are INVALID
   Fix: after mean-pooling, always call F.normalize(emb, p=2, dim=1).
   This is done in PatentBERTWrapper.encode() and _encode_text().

3. MODE-AWARE DECISION LOGIC
   ─────────────────────────────────────────────────
   Patent-BERT mode:
     • BERT is now the PRIMARY signal (higher weight learned from data)
     • Drift safeguard thresholds are TIGHTER (patent BERT is domain-adapted,
       so a high BERT score is meaningful, not an artefact)
     • DRIFT_TFIDF_MAX lowered back to 0.12 (only trigger on very low TF-IDF)
     • DRIFT_BERT_MIN  raised  to 0.85 (higher bar before calling "drift")
     • Domain coherence check still runs as a secondary guard

   General-BERT mode (USE_PATENT_BERT = False):
     • All v8 thresholds apply unchanged
     • DRIFT_TFIDF_MAX = 0.25, DRIFT_BERT_MIN = 0.72
     • Drift safeguard is the primary line of defence

4. FULL AUTO-RECALIBRATION AFTER MODEL SWITCH
   ─────────────────────────────────────────────────
   The cache key now includes the model name.
   Switching USE_PATENT_BERT automatically invalidates the cache and
   triggers a fresh compute_embeddings() + train_hybrid_model() cycle.
   You will NEVER silently run with stale embeddings from the wrong model.

5. EMBEDDING CACHE VERSIONING
   ─────────────────────────────────────────────────
   metadata.pkl now stores 'model_version' = sbert_model_name.
   load_cached_embeddings() checks this field; if it doesn't match the
   currently requested model, the cache is rejected and recomputed.

DECISION HIERARCHY v9:
=======================

Patent-BERT mode  (USE_PATENT_BERT = True):
  Rule 1 — Drift Safeguard (tight thresholds for domain-adapted model):
    If top_tfidf < 0.12 AND top_bert > 0.85  → ACCEPT
  Rule 2 — Domain Coherence:
    If median_top5_tfidf < 0.08 AND max_cal >= novelty_floor  → ACCEPT
  Rule 3 — TF-IDF Override:
    If max_tfidf < 0.10 AND max_hybrid < 0.45  → ACCEPT
  Rule 4 — Out-of-domain Floor:
    If max_cal < novelty_floor (0.40)  → ACCEPT
  Rule 5 — Dual-condition Rejection:
    REJECT iff max_cal >= threshold AND top3_avg >= threshold - 0.10
  Rule 6 — Confidence Band (unchanged)

General-BERT mode  (USE_PATENT_BERT = False):
  Rule 1 — Drift Safeguard (v8 thresholds):
    If top_tfidf < 0.25 AND top_bert > 0.72  → ACCEPT
  Rule 2 — Domain Coherence:
    If median_top5_tfidf < 0.10 AND max_cal >= novelty_floor  → ACCEPT
  Rule 3 — TF-IDF Override:
    If max_tfidf < 0.15 AND max_hybrid < 0.55  → ACCEPT
  Rule 4 — Out-of-domain Floor:
    If max_cal < novelty_floor (0.40)  → ACCEPT
  Rule 5 — Dual-condition Rejection (same)
  Rule 6 — Confidence Band (same)

CHANGES v9 vs v8:
==================
FIX-V9-1: USE_PATENT_BERT flag — single switch for dual-mode operation
FIX-V9-2: PatentBERTWrapper.encode() always returns L2-normalised embeddings
FIX-V9-3: _encode_text() re-normalises after any model (safety guarantee)
FIX-V9-4: Mode-aware threshold selection in predict_novelty()
FIX-V9-5: Cache versioning — model name stored in metadata, verified on load
FIX-V9-6: Cache invalidation on model switch (hash includes model name)
FIX-V9-7: Fusion weight bounds widen in patent-BERT mode (BERT allowed up to 0.75)
FIX-V9-8: Sparsity floor lifted to 0.48 in patent-BERT mode (tighter model = less leeway)
FIX-V9-9: Diagnostic output clearly labels which mode is active
"""

import sys
import os
import re
import time
import pickle
import hashlib
import warnings
import logging
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np

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
    print("WARNING: FAISS not installed — pip install faiss-cpu")

import torch
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("WARNING: sentence-transformers not installed — pip install sentence-transformers")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: transformers not installed — pip install transformers")

torch.set_grad_enabled(False)

for resource in ['corpora/stopwords', 'corpora/wordnet']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# ★  MAIN CONFIGURATION SWITCH  ★
#
# Set USE_PATENT_BERT = True  to use anferico/bert-for-patents
#     (requires: pip install protobuf sentencepiece)
# Set USE_PATENT_BERT = False to use BAAI/bge-base-en-v1.5
#     (no extra install needed; v8 drift safeguards apply)
#
# Switching this flag automatically invalidates the cache and
# triggers a full recompute + recalibration. (FIX-V9-5/6)
# ============================================================
USE_PATENT_BERT = True   # ← flip this to switch modes


# ============================================================
# MODEL PRIORITY LIST
# ============================================================
PATENT_BERT_MODEL   = "anferico/bert-for-patents"
GENERAL_BERT_MODEL  = "BAAI/bge-base-en-v1.5"
FALLBACK_BERT_MODEL = "all-MiniLM-L6-v2"

SBERT_MODEL_PRIORITY_PATENT = [
    (PATENT_BERT_MODEL,        "hf_bert"),   # primary — domain-adapted
    ("AI-Growth-Lab/PatentBERT","hf_bert"),  # secondary patent model
    (GENERAL_BERT_MODEL,       "sbert"),     # general fallback
    (FALLBACK_BERT_MODEL,      "sbert"),     # last resort
]

SBERT_MODEL_PRIORITY_GENERAL = [
    (GENERAL_BERT_MODEL,       "sbert"),
    (FALLBACK_BERT_MODEL,      "sbert"),
]

# ============================================================
# MODE-AWARE THRESHOLDS  (FIX-V9-4)
#
# Patent-BERT mode: tighter drift thresholds because a high BERT
#   score from a domain-adapted model IS meaningful.
#
# General-BERT mode: wider drift thresholds (v8 values) because
#   all-domain models inflate similarity for everything.
# ============================================================

THRESHOLDS = {
    # ── Patent-BERT mode ─────────────────────────────────────
    "patent": {
        "DRIFT_TFIDF_MAX":        0.12,   # only flag very low TF-IDF
        "DRIFT_BERT_MIN":         0.85,   # high bar before calling drift
        "COHERENCE_TFIDF_MEDIAN": 0.08,   # tighter coherence requirement
        "TFIDF_OVERRIDE_MAX":     0.10,
        "HYBRID_OVERRIDE_MAX":    0.45,
    },
    # ── General-BERT mode (v8 values, unchanged) ─────────────
    "general": {
        "DRIFT_TFIDF_MAX":        0.25,
        "DRIFT_BERT_MIN":         0.72,
        "COHERENCE_TFIDF_MEDIAN": 0.10,
        "TFIDF_OVERRIDE_MAX":     0.15,
        "HYBRID_OVERRIDE_MAX":    0.55,
    },
}


# ============================================================
# PATENT BERT WRAPPER  (FIX-V9-2)
#
# Key fix: encode() ALWAYS returns L2-normalised embeddings.
# anferico/bert-for-patents is a raw HF BERT encoder.
# Without normalisation cosine similarity is unstable and
# thresholds learned during training become invalid at inference.
# ============================================================

class PatentBERTWrapper:
    """
    Wraps a raw HuggingFace BERT encoder to expose the same
    .encode() interface as SentenceTransformer.

    Pipeline:
      tokenise → last_hidden_state → mean-pool → L2-normalise

    The L2 normalisation step is MANDATORY for stable cosine
    similarity when using raw BERT models not trained as SBERT.
    """
    def __init__(self, model_name, device):
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device     = device
        self.model_name = model_name
        print(f"   [PatentBERTWrapper] Loaded {model_name}")
        print(f"   [PatentBERTWrapper] Embeddings: mean-pool + L2-normalise")

    @staticmethod
    def _mean_pool(token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        return (torch.sum(token_embeddings * mask_expanded, 1) /
                torch.clamp(mask_expanded.sum(1), min=1e-9))

    def encode(self, sentences, convert_to_tensor=True, device=None,
               show_progress_bar=False, batch_size=32,
               normalize_embeddings=True):          # FIX-V9-2: always normalise
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Batches")
            except ImportError:
                pass

        for start in iterator:
            batch = sentences[start: start + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            emb = self._mean_pool(out.last_hidden_state, enc['attention_mask'])
            # ── L2 normalise (FIX-V9-2) ──────────────────────────
            if normalize_embeddings:
                emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu())

        result = torch.cat(all_embeddings, dim=0)
        return result if convert_to_tensor else result.numpy()


# ============================================================
# PLATT CALIBRATOR
# ============================================================

class PlattCalibrator:
    def __init__(self):
        self.lr     = LogisticRegression(C=1.0, max_iter=1000)
        self.fitted = False

    def fit(self, scores, labels):
        scores = np.array(scores, dtype=float)
        labels = np.array(labels)
        if len(np.unique(labels)) < 2:
            self.fitted = False
            return
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
    Patent Novelty Pre-Screening System — v9.

    Dual-mode: set USE_PATENT_BERT at module level.
      True  → anferico/bert-for-patents (domain-adapted, primary signal)
      False → BAAI/bge-base-en-v1.5    (general, drift-guarded, v8 rules)
    """

    def __init__(self, cache_dir='cache/', model_dir='models/',
                 use_patent_bert=USE_PATENT_BERT):

        self.use_patent_bert = use_patent_bert
        self._mode           = "patent" if use_patent_bert else "general"
        self._thr            = THRESHOLDS[self._mode]   # active threshold set

        self.stop_words  = set(stopwords.words('english'))
        self.lemmatizer  = WordNetLemmatizer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Device]: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"\n[Mode]: {'Patent-BERT' if use_patent_bert else 'General-BERT (+ safeguards)'}")

        self.vectorizer          = None
        self.tfidf_matrix        = None
        self.tfidf_matrix_dense  = None
        self.tfidf_cache         = None

        self.sbert_model         = None
        self.sbert_model_name    = None
        self.sbert_model_type    = None
        self.sbert_enabled       = SBERT_AVAILABLE or TRANSFORMERS_AVAILABLE
        self.patent_embeddings   = None
        self.sbert_index         = None
        self.patent_ids_ordered  = None
        self.id_to_index         = None

        # ── Fusion weights (FIX-V9-7) ────────────────────────
        # In patent-BERT mode BERT is trustworthy, allow up to 0.75 weight.
        # In general mode keep the v7/v8 range (max 0.55 for BERT).
        if use_patent_bert:
            self._w_tfidf_min, self._w_tfidf_max = 0.25, 0.55
        else:
            self._w_tfidf_min, self._w_tfidf_max = 0.45, 0.70
        self.w_tfidf = 0.40
        self.w_bert  = 0.60

        self.final_threshold          = 0.50
        self.novelty_floor            = 0.40
        self.tfidf_plausibility_floor = 0.12
        self.rrf_k                    = 60

        self.calibrator = PlattCalibrator()

        self.text_map                   = None
        self.title_map                  = None
        self.patents_df                 = None
        self.citation_set               = None
        self.citation_set_bidirectional = set()

        self.cache_dir = cache_dir
        self.model_dir = model_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.hash_file = os.path.join(cache_dir, 'dataset_hash.txt')

        self._using_general_bert = not use_patent_bert

    # ============================================================
    # UTILITIES
    # ============================================================

    def preprocess(self, text):
        if pd.isna(text) or not str(text).strip():
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s\-]', '', text)
        text = re.sub(r'-', ' ', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words
                 if w not in self.stop_words and len(w) > 2]
        return " ".join(words)

    def _model_tag(self):
        """Short string identifying the active model — used in cache hash."""
        return self.sbert_model_name or (
            PATENT_BERT_MODEL if self.use_patent_bert else GENERAL_BERT_MODEL)

    def compute_robust_dataset_hash(self, patents_df):
        """
        Hash = content hash + model tag.  (FIX-V9-6)
        Switching the model changes the hash → cache auto-invalidated.
        """
        h = self._model_tag()          # model name is part of the hash
        for pid in patents_df['patent_id'].values:
            txt = patents_df[patents_df['patent_id'] == pid]['clean_text'].values[0]
            h += f"{pid}:{hashlib.md5(txt.encode()).hexdigest()}"
        return hashlib.md5(h.encode()).hexdigest()

    def check_dataset_changed(self, patents_df):
        current = self.compute_robust_dataset_hash(patents_df)
        prev    = None
        changed = True
        if os.path.exists(self.hash_file):
            with open(self.hash_file) as f:
                prev = f.read().strip()
            changed = (current != prev)
        return changed, current, prev

    def save_dataset_hash(self, h):
        with open(self.hash_file, 'w') as f:
            f.write(h)
        logger.info(f"Dataset hash saved: {h[:8]}...")

    # ============================================================
    # FAISS
    # ============================================================

    def build_faiss_index(self, embeddings):
        if not FAISS_AVAILABLE:
            return None
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        logger.info(f"FAISS index: {index.ntotal} vectors")
        return index

    # ============================================================
    # DATASET
    # ============================================================

    def build_citation_dataset(self, patent_file, abstract_file, citation_file,
                               min_citations=2, max_patents=10000):
        print("=" * 70)
        print("BUILDING CITATION-AWARE DATASET")
        print("=" * 70)

        print("\n[1/5] Analyzing citation graph...")
        citing_counter = Counter()
        cited_counter  = Counter()

        chunk_iter  = pd.read_csv(citation_file, sep='\t', dtype=str, chunksize=500000)
        first_chunk = next(chunk_iter)
        cols = first_chunk.columns.tolist()

        patent_col, cited_col = None, None
        for col in cols:
            cl = col.lower()
            if 'citing' in cl or cl == 'patent_id':
                patent_col = col
            if 'cited' in cl or 'citation_patent_id' in cl:
                cited_col = col
        if patent_col is None or cited_col is None:
            patent_col = cols[0]
            cited_col  = cols[2] if len(cols) > 2 else cols[1]
        print(f"Columns: citing='{patent_col}', cited='{cited_col}'")

        all_chunks = [first_chunk] + list(
            pd.read_csv(citation_file, sep='\t', dtype=str,
                        usecols=[patent_col, cited_col], chunksize=500000))
        for chunk in all_chunks:
            for v in chunk[patent_col].astype(str).fillna('').tolist():
                if v and v != 'nan': citing_counter[v] += 1
            for v in chunk[cited_col].astype(str).fillna('').tolist():
                if v and v != 'nan': cited_counter[v] += 1

        core = list(
            {p for p, c in cited_counter.items()  if c >= min_citations} &
            {p for p, c in citing_counter.items() if c >= min_citations})
        print(f"Core patents: {len(core):,}")
        if len(core) > max_patents:
            core = np.random.choice(core, max_patents, replace=False).tolist()
            print(f"Sampled {max_patents:,}")

        print("\n[2/5] Loading patent data...")
        chunks = []
        for chunk in pd.read_csv(patent_file, sep='\t', dtype=str, chunksize=10000):
            if 'patent_id' in chunk.columns:
                m = chunk['patent_id'].isin(core)
                if m.any(): chunks.append(chunk[m])
        patents = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(patents):,}")

        print("\n[3/5] Loading abstracts...")
        pid_set = set(patents['patent_id'].astype(str))
        abs_chunks = []
        for chunk in pd.read_csv(abstract_file, sep='\t', dtype=str, chunksize=10000):
            if 'patent_id' in chunk.columns:
                m = chunk['patent_id'].astype(str).isin(pid_set)
                if m.any(): abs_chunks.append(chunk[m])
        abstracts = (pd.concat(abs_chunks, ignore_index=True)
                     if abs_chunks else pd.DataFrame())

        df = (patents.merge(abstracts, on='patent_id', how='inner')
              if len(abstracts) > 0 else patents.copy())
        if len(abstracts) == 0:
            df['patent_abstract'] = ""

        title_col, abs_col = None, None
        for col in df.columns:
            if 'title'    in col.lower(): title_col = col
            if 'abstract' in col.lower(): abs_col   = col
        if title_col is None: title_col = df.columns[1]
        if abs_col   is None: abs_col   = title_col

        df = df[['patent_id', title_col, abs_col]].dropna()
        df.columns = ['patent_id', 'patent_title', 'patent_abstract']
        print(f"Patent count: {len(df):,}")

        print("\n[4/5] Preprocessing...")
        df['clean_text'] = (df['patent_title'] + " " + df['patent_abstract']
                            ).apply(self.preprocess)
        df = df[df['clean_text'].str.split().str.len() >= 5].reset_index(drop=True)
        print(f"After filter: {len(df):,}")

        self.text_map  = dict(zip(df['patent_id'], df['clean_text']))
        self.title_map = dict(zip(df['patent_id'], df['patent_title']))

        print("\n[5/5] Extracting citation pairs...")
        valid  = set(df['patent_id'].astype(str))
        cpairs = []
        for chunk in pd.read_csv(citation_file, sep='\t', dtype=str,
                                 usecols=[patent_col, cited_col], chunksize=500000):
            chunk = chunk.rename(
                columns={patent_col: 'patent_id', cited_col: 'cited_patent_id'})
            chunk['patent_id']       = chunk['patent_id'].astype(str)
            chunk['cited_patent_id'] = chunk['cited_patent_id'].astype(str)
            m = (chunk['patent_id'].isin(valid) &
                 chunk['cited_patent_id'].isin(valid) &
                 (chunk['patent_id'] != chunk['cited_patent_id']))
            if m.any(): cpairs.append(chunk[m])

        citations = (pd.concat(cpairs, ignore_index=True)
                     if cpairs else pd.DataFrame())
        print(f"Citation pairs: {len(citations):,}")

        n = len(df)
        density = len(citations) / max(n * (n - 1), 1)
        self._sparse_dataset = density < 0.0001
        if self._sparse_dataset:
            print(f"   [WARNING] Low citation density ({density:.6%}) — threshold adj applied")

        if len(citations) == 0:
            return self._create_demo_data()

        self.citation_set = set(zip(citations['patent_id'],
                                    citations['cited_patent_id']))
        self.citation_set_bidirectional = (
            self.citation_set | {(b, a) for a, b in self.citation_set})

        pos = citations.sample(
            n=min(1500, len(citations)), random_state=42).copy()
        pos['label'] = 1
        print(f"Positive pairs: {len(pos)}")

        self.patent_ids_ordered = list(valid)
        self.patents_df = df
        return df, pos

    def _create_demo_data(self, num_patents=300):
        print(f"\nCreating demo dataset ({num_patents} patents)...")
        self._sparse_dataset = False
        domains = {
            'neural_networks': [
                'neural network','deep learning','backpropagation',
                'LSTM','transformer','attention','gradient descent','convolutional'],
            'computer_vision': [
                'object detection','image segmentation','face recognition',
                'convolution','feature extraction','bounding box','pixel'],
            'nlp': [
                'text classification','sentiment analysis','machine translation',
                'language model','tokenization','named entity','BERT'],
            'reinforcement_learning': [
                'Q-learning','policy gradient','deep Q network',
                'actor-critic','reward function','Markov','exploration'],
            'optimization': [
                'gradient descent','Adam optimizer','learning rate',
                'regularization','hyperparameter','loss function','convergence'],
        }
        patents, pid = [], 1
        for domain, terms in domains.items():
            for _ in range(num_patents // len(domains)):
                main = np.random.choice(terms)
                sel  = np.random.choice(terms,
                                        size=min(5, len(terms)), replace=False)
                patents.append({
                    'patent_id':       f"PAT{pid:04d}",
                    'patent_title':    (f"System for {main.lower()} "
                                        f"in {domain.replace('_',' ')}"),
                    'patent_abstract': (f"A {domain.replace('_',' ')} approach using "
                                        f"{', '.join(sel[:-1])}, and {sel[-1]}. "
                                        f"Improves over prior art in "
                                        f"{domain.replace('_',' ')}.")
                })
                pid += 1

        df = pd.DataFrame(patents)
        df['clean_text'] = (df['patent_title'] + " " + df['patent_abstract']
                            ).apply(self.preprocess)
        self.text_map  = dict(zip(df['patent_id'], df['clean_text']))
        self.title_map = dict(zip(df['patent_id'], df['patent_title']))
        self.patents_df = df
        self.patent_ids_ordered = df['patent_id'].tolist()

        dmap = {}
        for p in patents:
            for d in domains:
                if d.replace('_', ' ') in p['patent_abstract']:
                    dmap[p['patent_id']] = d
                    break

        cpairs = []
        pids = df['patent_id'].tolist()
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                if (dmap.get(pids[i]) == dmap.get(pids[j]) and
                        np.random.random() < 0.25):
                    cpairs.append((pids[i], pids[j]))

        cdf = pd.DataFrame(cpairs, columns=['patent_id', 'cited_patent_id'])
        self.citation_set = set(zip(cdf['patent_id'], cdf['cited_patent_id']))
        self.citation_set_bidirectional = (
            self.citation_set | {(b, a) for a, b in self.citation_set})

        pos = cdf.sample(n=min(500, len(cdf)), random_state=42).copy()
        pos['label'] = 1
        print(f"Demo: {len(df)} patents, {len(cdf)} pairs")
        return df, pos

    # ============================================================
    # SCORE HELPERS
    # ============================================================

    def _tfidf_sim(self, p1, p2):
        return float(np.clip(
            cosine_similarity(
                self.tfidf_cache[p1], self.tfidf_cache[p2])[0][0],
            0.0, 1.0))

    def _bert_sim(self, i1, i2):
        """
        Dot product of L2-normalised embeddings = cosine similarity.
        Range already [0,1] after normalisation when embeddings are
        non-negative (which mean-pooled BERT vectors usually are).
        We still clip to be safe.
        """
        raw = float(torch.dot(self.patent_embeddings[i1],
                              self.patent_embeddings[i2]).item())
        # L2-normalised vectors: dot ∈ [-1,1].  Map to [0,1].
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    def _fuse(self, tfidf, bert):
        """
        Plausibility-capped fusion.
        If TF-IDF is below the plausibility floor, scale BERT down
        proportionally so BERT alone cannot drive a rejection.
        In patent-BERT mode this floor is lower (0.08) because the
        model's semantic signal is more trustworthy.
        """
        floor = (0.08 if self.use_patent_bert
                 else self.tfidf_plausibility_floor)
        if tfidf < floor:
            bert = bert * (tfidf / floor)
        return float(np.clip(
            self.w_tfidf * tfidf + self.w_bert * bert, 0.0, 1.0))

    def _compute_pair_score_direct(self, p1, p2):
        i1 = self.id_to_index.get(str(p1))
        i2 = self.id_to_index.get(str(p2))
        if i1 is None or i2 is None:
            return None, None
        t   = self._tfidf_sim(str(p1), str(p2))
        b   = self._bert_sim(i1, i2)
        raw = self._fuse(t, b)
        cal = (float(self.calibrator.predict_proba(np.array([raw]))[0])
               if self.calibrator.fitted else raw)
        return raw, cal

    # ============================================================
    # MODEL LOADING  (FIX-V9-1)
    # ============================================================

    def _load_sbert_model(self):
        if not (SBERT_AVAILABLE or TRANSFORMERS_AVAILABLE):
            raise RuntimeError(
                "Install sentence-transformers or transformers.")

        priority = (SBERT_MODEL_PRIORITY_PATENT if self.use_patent_bert
                    else SBERT_MODEL_PRIORITY_GENERAL)

        for model_name, method in priority:
            try:
                print(f"\n   Trying: {model_name} (method={method})")

                if method == 'hf_bert':
                    if not TRANSFORMERS_AVAILABLE:
                        print(f"   Skipped (transformers not installed)")
                        continue
                    model = PatentBERTWrapper(model_name, self.device)
                    # Smoke test — also confirms normalisation works
                    test_emb = model.encode("patent claim",
                                            convert_to_tensor=False,
                                            normalize_embeddings=True)
                    norm = float(np.linalg.norm(test_emb))
                    print(f"   Smoke test embedding norm: {norm:.4f} "
                          f"(should be ~1.0 after normalisation)")
                    self.sbert_model_name    = model_name
                    self.sbert_model_type    = 'hf_bert'
                    self._using_general_bert = False
                    print(f"   ✓ Patent BERT loaded: {model_name}")
                    return model

                else:  # sbert
                    if not SBERT_AVAILABLE:
                        print(f"   Skipped (sentence-transformers not installed)")
                        continue
                    model = SentenceTransformer(model_name)
                    model = model.to(self.device)
                    # SBERT models already normalise by default in newer versions
                    # but we re-normalise in _encode_text() as safety (FIX-V9-3)
                    test_emb = model.encode("patent claim",
                                            convert_to_tensor=False,
                                            normalize_embeddings=True)
                    norm = float(np.linalg.norm(test_emb))
                    print(f"   Smoke test embedding norm: {norm:.4f}")
                    self.sbert_model_name    = model_name
                    self.sbert_model_type    = 'sbert'
                    self._using_general_bert = model_name in (
                        GENERAL_BERT_MODEL, FALLBACK_BERT_MODEL)
                    status = ("⚠ General-purpose model"
                              if self._using_general_bert
                              else "✓ Patent model")
                    print(f"   {status} loaded: {model_name}")
                    if self._using_general_bert:
                        print(f"   NOTE: v9 drift safeguards (general mode) active.")
                        print(f"   TIP:  pip install protobuf sentencepiece  "
                              f"→ re-run for patent BERT")
                    return model

            except Exception as e:
                msg = str(e)[:120]
                if 'protobuf' in msg.lower() or 'sentencepiece' in msg.lower():
                    print(f"   ✗ {model_name} — missing lib: "
                          f"pip install protobuf sentencepiece")
                elif '401' in msg or 'unauthorized' in msg.lower():
                    print(f"   ✗ {model_name} — private repo: "
                          f"set HF_TOKEN or run huggingface-cli login")
                else:
                    print(f"   ✗ {model_name} — {e.__class__.__name__}: {msg}")
                continue

        raise RuntimeError("All BERT models failed to load.")

    def _encode_text(self, text_or_list, show_progress_bar=False, batch_size=32):
        """
        Encode text and ALWAYS return L2-normalised embeddings. (FIX-V9-3)
        This is a safety guarantee: even if a model does not normalise
        internally, the output from this function is always unit-norm.
        """
        is_str = isinstance(text_or_list, str)
        if is_str:
            text_or_list = [text_or_list]

        # Pass normalize_embeddings=True to both wrappers.
        # PatentBERTWrapper honours it explicitly.
        # SentenceTransformer honours it natively (≥2.2.0).
        kwargs = dict(
            convert_to_tensor=True,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            normalize_embeddings=True,       # FIX-V9-2 / V9-3
        )
        if self.sbert_model_type == 'sbert':
            kwargs['device'] = self.device

        emb = self.sbert_model.encode(text_or_list, **kwargs)

        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        emb = emb.to(self.device)

        # Re-normalise as a hard safety guarantee (FIX-V9-3)
        emb = F.normalize(emb, p=2, dim=1)
        return emb[0] if is_str else emb

    # ============================================================
    # EMBEDDINGS
    # ============================================================

    def compute_embeddings(self):
        print("\n" + "=" * 70)
        print("COMPUTING EMBEDDINGS")
        print("=" * 70)

        print("\n[1/2] TF-IDF (trigrams, 15k features)...")
        self.vectorizer = TfidfVectorizer(
            max_features=15000, ngram_range=(1, 3),
            min_df=2, max_df=0.85, sublinear_tf=True)
        self.vectorizer.fit(list(self.text_map.values()))
        self.patent_ids_ordered = self.patents_df['patent_id'].tolist()

        from scipy.sparse import vstack
        tlist = [self.vectorizer.transform([self.text_map[p]])
                 for p in self.patent_ids_ordered]
        self.tfidf_matrix = vstack(tlist)
        self.tfidf_cache  = dict(zip(self.patent_ids_ordered, tlist))

        n = len(self.patent_ids_ordered)
        if n <= 15000:
            self.tfidf_matrix_dense = self.tfidf_matrix.toarray().astype('float32')
            norms = np.linalg.norm(
                self.tfidf_matrix_dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.tfidf_matrix_dense /= norms
            print(f"   Dense TF-IDF cached: {self.tfidf_matrix_dense.shape}")
        else:
            self.tfidf_matrix_dense = None

        if self.sbert_enabled:
            print("\n[2/2] BERT embeddings...")
            if self.sbert_model is None:
                self.sbert_model = self._load_sbert_model()

            texts = [self.text_map[p] for p in self.patent_ids_ordered]
            self.patent_embeddings = self._encode_text(
                texts, show_progress_bar=True, batch_size=32)

            # Verify normalisation
            norms = torch.norm(self.patent_embeddings, dim=1)
            print(f"   Embedding norms — mean: {norms.mean():.4f}  "
                  f"std: {norms.std():.4f}  (should be ~1.000 ± 0.001)")

            self.id_to_index = {
                p: i for i, p in enumerate(self.patent_ids_ordered)}

            if FAISS_AVAILABLE:
                np_emb = (self.patent_embeddings.cpu()
                          .numpy().astype('float32'))
                self.sbert_index = self.build_faiss_index(np_emb)
        else:
            self.id_to_index = {
                p: i for i, p in enumerate(self.patent_ids_ordered)}

        print("\nEmbeddings done!")
        self.save_cached_embeddings()

    def init_sbert(self):
        if not (SBERT_AVAILABLE or TRANSFORMERS_AVAILABLE):
            self.sbert_enabled = False
            return False
        if self.sbert_model is not None:
            return True
        try:
            self.sbert_model  = self._load_sbert_model()
            self.sbert_enabled = True
            return True
        except Exception as e:
            print(f"  All models failed: {e}")
            self.sbert_enabled = False
            return False

    # ============================================================
    # NEGATIVE MINING
    # ============================================================

    def get_random_negatives(self, query_ids, target, rng=None):
        rng = rng or np.random.RandomState(99)
        negatives, attempts = [], 0
        while len(negatives) < target and attempts < target * 20:
            attempts += 1
            p1 = rng.choice(query_ids)
            p2 = rng.choice(self.patent_ids_ordered)
            if p1 == p2 or (p1, p2) in self.citation_set_bidirectional:
                continue
            negatives.append({'patent_id': p1, 'cited_patent_id': p2, 'label': 0})
        return pd.DataFrame(negatives)

    def get_medium_hard_negatives(self, positive_pairs, target, rng=None):
        rng = rng or np.random.RandomState(42)
        print("\n   Mining medium-hard negatives (10th–40th pct TF-IDF)...")
        negatives, attempts = [], 0
        qids = positive_pairs['patent_id'].unique().tolist()
        while len(negatives) < target and attempts < target * 20:
            attempts += 1
            p1   = rng.choice(qids)
            sims = cosine_similarity(
                self.tfidf_cache[p1], self.tfidf_matrix)[0]
            idx  = np.argsort(sims)[::-1]
            nn   = len(idx)
            lo, hi = int(nn * 0.10), int(nn * 0.40)
            pool = idx[lo:hi]
            if not len(pool): continue
            p2 = self.patent_ids_ordered[rng.choice(pool)]
            if p1 == p2 or (p1, p2) in self.citation_set_bidirectional:
                continue
            negatives.append({'patent_id': p1, 'cited_patent_id': p2, 'label': 0})
        df = pd.DataFrame(negatives)
        print(f"   Mined {len(df)} medium-hard negatives")
        return df

    def get_mixed_negatives(self, positive_pairs, target,
                            hard_ratio=0.65, rng=None):
        rng    = rng or np.random.RandomState(42)
        n_hard = int(target * hard_ratio)
        n_easy = target - n_hard
        hard   = self.get_medium_hard_negatives(positive_pairs, n_hard, rng=rng)
        easy   = self.get_random_negatives(
            positive_pairs['patent_id'].unique().tolist(), n_easy, rng=rng)
        mixed  = pd.concat([hard, easy], ignore_index=True)
        mixed['label'] = 0
        return mixed

    # ============================================================
    # TRAINING
    # ============================================================

    def train_hybrid_model(self, positive_pairs):
        print("\n" + "=" * 70)
        print(f"TRAINING HYBRID MODEL  [{self._mode.upper()} mode]")
        print("=" * 70)

        if self.tfidf_matrix is None:
            self.compute_embeddings()

        pos   = positive_pairs.sample(frac=1, random_state=42
                                      ).reset_index(drop=True)
        n_val = max(60, int(len(pos) * 0.30))
        val_pos   = pos.iloc[:n_val].copy()
        train_pos = pos.iloc[n_val:].copy()
        print(f"\n   Train+: {len(train_pos)} | Val+: {len(val_pos)}")

        print("\n[1/4] Training positives...")
        tt, bt, yl = [], [], []
        for _, row in train_pos.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1, i2 = self.id_to_index.get(p1), self.id_to_index.get(p2)
            if i1 is None or i2 is None: continue
            tt.append(self._tfidf_sim(p1, p2))
            bt.append(self._bert_sim(i1, i2))
            yl.append(1)
        n_pos = len(yl)
        print(f"   Valid: {n_pos}")

        print("\n[2/4] Mining negatives...")
        neg_df = self.get_mixed_negatives(
            train_pos, n_pos, 0.65, np.random.RandomState(42))
        for _, row in neg_df.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1, i2 = self.id_to_index.get(p1), self.id_to_index.get(p2)
            if i1 is None or i2 is None: continue
            tt.append(self._tfidf_sim(p1, p2))
            bt.append(self._bert_sim(i1, i2))
            yl.append(0)
        print(f"\n   Total: {len(yl)}  "
              f"(pos={sum(yl)}, neg={len(yl)-sum(yl)})")

        print("\n[3/4] Learning fusion weights...")
        X  = np.column_stack([tt, bt])
        y  = np.array(yl)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X, y)
        raw_w = lr.coef_[0]
        ws    = np.sum(np.abs(raw_w))
        if ws > 0:
            lw = np.abs(raw_w) / ws
            # FIX-V9-7: mode-aware weight bounds
            self.w_tfidf = float(
                np.clip(lw[0], self._w_tfidf_min, self._w_tfidf_max))
            self.w_bert  = 1.0 - self.w_tfidf
        print(f"   Mode:    {self._mode}")
        print(f"   Weights: TF-IDF={self.w_tfidf:.3f}, BERT={self.w_bert:.3f}")
        print(f"   (Patent-BERT mode allows BERT weight up to "
              f"{1.0 - self._w_tfidf_min:.2f})")

        fused = np.array([self._fuse(t, b) for t, b in zip(tt, bt)])
        pm    = y == 1
        print(f"   Score sep (fused): "
              f"pos={fused[pm].mean():.3f} "
              f"neg={fused[~pm].mean():.3f} "
              f"gap={fused[pm].mean()-fused[~pm].mean():.3f}")

        print("\n[4/4] Validation + calibration + threshold...")
        vn_df   = self.get_mixed_negatives(
            val_pos, len(val_pos), 0.65, np.random.RandomState(77))
        val_all = pd.concat([val_pos, vn_df], ignore_index=True)

        vt, vb, vl = [], [], []
        for _, row in val_all.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1, i2 = self.id_to_index.get(p1), self.id_to_index.get(p2)
            if i1 is None or i2 is None: continue
            vt.append(self._tfidf_sim(p1, p2))
            vb.append(self._bert_sim(i1, i2))
            vl.append(int(row['label']))

        vf  = np.array([self._fuse(t, b) for t, b in zip(vt, vb)])
        vla = np.array(vl)
        print(f"\n   Val: {len(vla)} "
              f"(pos={vla.sum()}, neg={(vla==0).sum()})")

        self.calibrator.fit(vf, vla)
        print(f"   Calibrator fitted: {self.calibrator.fitted}")

        vp = self.calibrator.predict_proba(vf) if self.calibrator.fitted else vf
        print(f"   Calibrated range: [{vp.min():.3f}, {vp.max():.3f}]")
        if vla.sum() > 0:
            print(f"     Pos mean: {vp[vla==1].mean():.3f}")
        if (vla == 0).sum() > 0:
            print(f"     Neg mean: {vp[vla==0].mean():.3f}")

        self._optimize_threshold(vp, vla)

        # FIX-V9-8: sparsity floor — tighter in patent-BERT mode
        if getattr(self, '_sparse_dataset', False):
            sparsity_floor = 0.48 if self.use_patent_bert else 0.45
            old = self.final_threshold
            self.final_threshold = max(sparsity_floor,
                                       self.final_threshold - 0.03)
            print(f"\n   [Sparsity adj] {old:.3f} → {self.final_threshold:.3f} "
                  f"(floor={sparsity_floor})")

        self.save_cached_embeddings()
        return True

    def _optimize_threshold(self, y_scores, y_true):
        print(f"\n   Optimizing threshold (BAC)...")
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        if n_pos == 0 or n_neg == 0:
            self.final_threshold = 0.50
            return

        best_bac, best_t, best_f1 = -1.0, 0.50, 0.0
        for t in np.linspace(0.15, 0.85, 281):
            preds = (y_scores >= t).astype(int)
            bac   = balanced_accuracy_score(y_true, preds)
            f1    = f1_score(y_true, preds, zero_division=0)
            if bac > best_bac or (bac == best_bac and f1 > best_f1):
                best_bac, best_t, best_f1 = bac, t, f1

        self.final_threshold = max(0.50, float(best_t))

        preds = (y_scores >= self.final_threshold).astype(int)
        print(f"   Learned={best_t:.3f}  Applied={self.final_threshold:.3f} "
              f"(floor=0.50)")
        print(f"   Acc={accuracy_score(y_true,preds):.3f}  "
              f"Prec={precision_score(y_true,preds,zero_division=0):.3f}  "
              f"Rec={recall_score(y_true,preds,zero_division=0):.3f}  "
              f"F1={f1_score(y_true,preds,zero_division=0):.3f}  "
              f"BAC={balanced_accuracy_score(y_true,preds):.3f}")
        if n_pos > 0 and n_neg > 0:
            print(f"   AUC={roc_auc_score(y_true, y_scores):.3f}")

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def _tfidf_top(self, qvec, n):
        if self.tfidf_matrix_dense is not None:
            qd = qvec.toarray().astype('float32').flatten()
            qn = np.linalg.norm(qd)
            if qn > 0: qd /= qn
            sims = self.tfidf_matrix_dense @ qd
        else:
            sims = cosine_similarity(qvec, self.tfidf_matrix)[0]
        sims = np.clip(sims, 0.0, 1.0)
        idx  = np.argsort(sims)[::-1][:n]
        return idx, sims[idx]

    def _rrf(self, bert_idx, tfidf_idx, k=60):
        scores = {}
        for rank, idx in enumerate(bert_idx):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(tfidf_idx):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def compute_hybrid_similarity(self, query_text, top_k=100):
        if not self.sbert_enabled or self.patent_embeddings is None:
            return self._tfidf_only(query_text, top_k)

        clean = self.preprocess(query_text)
        if not clean.strip():
            return []

        n_bert  = min(top_k * 6, len(self.patent_ids_ordered))
        n_tfidf = min(top_k * 8, len(self.patent_ids_ordered))

        # TF-IDF retrieval
        qvec      = self.vectorizer.transform([clean])
        tfidf_top_idx, _ = self._tfidf_top(qvec, n_tfidf)
        tfidf_all = np.clip(
            cosine_similarity(qvec, self.tfidf_matrix)[0], 0.0, 1.0)

        # BERT retrieval
        qemb = self._encode_text(clean)   # always L2-normalised (FIX-V9-3)
        if self.sbert_index is not None and FAISS_AVAILABLE:
            qnp = qemb.cpu().numpy().reshape(1, -1).astype('float32')
            faiss.normalize_L2(qnp)
            bscores_raw, bidx = self.sbert_index.search(qnp, n_bert)
            bert_top_idx  = bidx[0]
            bert_scores_d = {
                int(i): float(np.clip((s + 1) / 2, 0, 1))
                for i, s in zip(bert_top_idx, bscores_raw[0])}
        else:
            braw = torch.mm(
                qemb.unsqueeze(0), self.patent_embeddings.T)[0].cpu().numpy()
            ball = np.clip((braw + 1.0) / 2.0, 0.0, 1.0)
            bert_top_idx  = np.argsort(ball)[::-1][:n_bert]
            bert_scores_d = {int(i): float(ball[i]) for i in bert_top_idx}

        rrf = self._rrf(bert_top_idx, tfidf_top_idx, k=self.rrf_k)

        results = []
        for orig_idx, _ in rrf:
            pid     = self.patent_ids_ordered[orig_idx]
            tfidf_s = float(tfidf_all[orig_idx])
            bert_s  = bert_scores_d.get(
                orig_idx,
                float(np.clip(
                    (torch.dot(qemb,
                               self.patent_embeddings[orig_idx]).item() + 1) / 2,
                    0, 1)))
            hybrid  = self._fuse(tfidf_s, bert_s)
            cal     = (float(self.calibrator.predict_proba(np.array([hybrid]))[0])
                       if self.calibrator.fitted else hybrid)
            results.append({
                'patent_id':        pid,
                'title':            self.title_map.get(pid, pid),
                'tfidf_sim':        tfidf_s,
                'bert_sim':         bert_s,
                'hybrid_sim':       hybrid,
                'calibrated_score': cal,
            })

        results.sort(key=lambda x: x['hybrid_sim'], reverse=True)
        return results[:top_k]

    def _tfidf_only(self, query_text, top_k=100):
        clean = self.preprocess(query_text)
        qvec  = self.vectorizer.transform([clean])
        sims  = np.clip(
            cosine_similarity(qvec, self.tfidf_matrix)[0], 0.0, 1.0)
        idx   = np.argsort(sims)[::-1][:top_k]
        return [{
            'patent_id':        self.patent_ids_ordered[i],
            'title':            self.title_map.get(
                                    self.patent_ids_ordered[i], ''),
            'tfidf_sim':        float(sims[i]),
            'bert_sim':         float(sims[i]),
            'hybrid_sim':       float(sims[i]),
            'calibrated_score': float(sims[i]),
        } for i in idx]

    # ============================================================
    # EVALUATION
    # ============================================================

    def evaluate_model(self, eval_pairs, eval_top_k=100):
        if eval_pairs is None or len(eval_pairs) == 0:
            print("No eval pairs.")
            return None

        print("\n" + "=" * 70)
        print(f"MODEL EVALUATION  [{self._mode.upper()} mode]")
        print("=" * 70)

        y_true, y_raw, y_cal = [], [], []
        rcounts = {10: 0, 20: 0, 50: 0, 100: 0}
        n_pos_total, skipped = 0, 0

        for _, row in eval_pairs.iterrows():
            label = int(row['label'])
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            raw, cal = self._compute_pair_score_direct(p1, p2)
            if raw is None:
                skipped += 1
                continue

            if label == 1:
                n_pos_total += 1
                qtxt = self.text_map.get(p1, '')
                if qtxt:
                    res  = self.compute_hybrid_similarity(
                        qtxt, top_k=eval_top_k)
                    pids = [str(r['patent_id']) for r in res]
                    for k in rcounts:
                        if p2 in pids[:k]:
                            rcounts[k] += 1

            y_true.append(label)
            y_raw.append(float(raw))
            y_cal.append(float(cal))

        if skipped:
            print(f"   Skipped {skipped}")

        y_true = np.array(y_true)
        y_raw  = np.array(y_raw)
        y_cal  = np.array(y_cal)
        n_p    = int(y_true.sum())
        n_n    = len(y_true) - n_p

        print(f"\n   Eval: {len(y_true)} (pos={n_p}, neg={n_n})")
        mode_str = ("✓ Patent-specific" if not self._using_general_bert
                    else "⚠ General-purpose")
        print(f"   Model: {self.sbert_model_name}  {mode_str}")

        print(f"\n   RAW Hybrid:")
        if n_p:
            print(f"     Pos: {y_raw[y_true==1].mean():.3f} "
                  f"± {y_raw[y_true==1].std():.3f}")
        if n_n:
            print(f"     Neg: {y_raw[y_true==0].mean():.3f} "
                  f"± {y_raw[y_true==0].std():.3f}")
        if n_p and n_n:
            print(f"     Sep: "
                  f"{y_raw[y_true==1].mean()-y_raw[y_true==0].mean():.3f}")

        print(f"\n   CALIBRATED:")
        if n_p:
            print(f"     Pos: {y_cal[y_true==1].mean():.3f} "
                  f"± {y_cal[y_true==1].std():.3f}")
        if n_n:
            print(f"     Neg: {y_cal[y_true==0].mean():.3f} "
                  f"± {y_cal[y_true==0].std():.3f}")
        if n_p and n_n:
            print(f"     Sep: "
                  f"{y_cal[y_true==1].mean()-y_cal[y_true==0].mean():.3f}")

        ypred = (y_cal >= self.final_threshold).astype(int)
        auc   = (roc_auc_score(y_true, y_cal)
                 if n_p and n_n else float('nan'))
        ap    = (average_precision_score(y_true, y_cal)
                 if n_p and n_n else float('nan'))

        metrics = {
            'accuracy':          accuracy_score(y_true, ypred),
            'balanced_accuracy': balanced_accuracy_score(y_true, ypred),
            'precision':         precision_score(y_true, ypred,
                                                 zero_division=0),
            'recall':            recall_score(y_true, ypred,
                                              zero_division=0),
            'f1':                f1_score(y_true, ypred, zero_division=0),
            'auc_roc':           auc,
            'avg_precision':     ap,
            'threshold':         self.final_threshold,
        }
        for k in rcounts:
            metrics[f'recall_at_{k}'] = rcounts[k] / max(n_pos_total, 1)

        print(f"\n[Classification] (threshold={self.final_threshold:.3f}):")
        print(f"   Accuracy:   {metrics['accuracy']:.3f}  "
              f"({metrics['accuracy']*100:.1f}%)")
        print(f"   Bal Acc:    {metrics['balanced_accuracy']:.3f}")
        print(f"   Precision:  {metrics['precision']:.3f}")
        print(f"   Recall:     {metrics['recall']:.3f}")
        print(f"   F1:         {metrics['f1']:.3f}")
        if not np.isnan(auc):
            print(f"   AUC-ROC:    {auc:.3f}")
        if not np.isnan(ap):
            print(f"   Avg Prec:   {ap:.3f}")

        print(f"\n[Retrieval] ({n_pos_total} positives):")
        for k in sorted(rcounts):
            r = metrics[f'recall_at_{k}']
            print(f"   Recall@{k:<4}: {r:.3f}  ({r*100:.1f}%)")

        return metrics

    # ============================================================
    # CACHE  (FIX-V9-5)
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
        if self.tfidf_matrix_dense is not None:
            np.save(os.path.join(self.cache_dir, 'tfidf_dense.npy'),
                    self.tfidf_matrix_dense)
        meta = {
            # FIX-V9-5: model name stored for version check on load
            'model_version':            self.sbert_model_name,
            'patent_ids_ordered':       self.patent_ids_ordered,
            'title_map':                self.title_map,
            'text_map':                 self.text_map,
            'citation_set':             (list(self.citation_set)
                                         if self.citation_set else []),
            'w_tfidf':                  self.w_tfidf,
            'w_bert':                   self.w_bert,
            'final_threshold':          self.final_threshold,
            'novelty_floor':            self.novelty_floor,
            'tfidf_plausibility_floor': self.tfidf_plausibility_floor,
            'rrf_k':                    self.rrf_k,
            'calibrator':               self.calibrator,
            'sparse_dataset':           getattr(self, '_sparse_dataset', False),
            'sbert_model_name':         self.sbert_model_name,
            'sbert_model_type':         self.sbert_model_type,
            'using_general_bert':       self._using_general_bert,
            'use_patent_bert':          self.use_patent_bert,
        }
        with open(os.path.join(self.cache_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        logger.info("Cache saved")

    def load_cached_embeddings(self):
        """
        Load embeddings from cache.
        FIX-V9-5: Reject cache if it was built with a different model.
        """
        from scipy.sparse import load_npz
        paths = {
            'tfidf': os.path.join(self.cache_dir, 'tfidf_matrix.npz'),
            'bert':  os.path.join(self.cache_dir, 'bert_embeddings.pt'),
            'vec':   os.path.join(self.cache_dir, 'vectorizer.pkl'),
            'meta':  os.path.join(self.cache_dir, 'metadata.pkl'),
            'dense': os.path.join(self.cache_dir, 'tfidf_dense.npy'),
        }
        if not all(os.path.exists(paths[k])
                   for k in ['tfidf', 'vec', 'meta']):
            return False
        try:
            with open(paths['meta'], 'rb') as f:
                m = pickle.load(f)

            # FIX-V9-5: model version check
            cached_model = m.get('model_version', None)
            expected_model = (PATENT_BERT_MODEL if self.use_patent_bert
                              else GENERAL_BERT_MODEL)
            if cached_model != expected_model:
                logger.warning(
                    f"Cache model mismatch: cached='{cached_model}' "
                    f"expected='{expected_model}' — recomputing")
                return False

            self.tfidf_matrix = load_npz(paths['tfidf'])
            if os.path.exists(paths['dense']):
                self.tfidf_matrix_dense = np.load(paths['dense'])
            if self.sbert_enabled and os.path.exists(paths['bert']):
                self.patent_embeddings = torch.load(
                    paths['bert'], map_location='cpu')
                if self.device.type == 'cuda':
                    self.patent_embeddings = (
                        self.patent_embeddings.to(self.device))
            with open(paths['vec'], 'rb') as f:
                self.vectorizer = pickle.load(f)

            self.patent_ids_ordered       = m['patent_ids_ordered']
            self.title_map                = m['title_map']
            self.text_map                 = m['text_map']
            self.citation_set             = set(m.get('citation_set', []))
            self.citation_set_bidirectional = (
                self.citation_set |
                {(b, a) for a, b in self.citation_set})
            self.w_tfidf                  = m.get('w_tfidf', 0.40)
            self.w_bert                   = m.get('w_bert',  0.60)
            self.final_threshold          = m.get('final_threshold', 0.50)
            self.novelty_floor            = m.get('novelty_floor', 0.40)
            self.tfidf_plausibility_floor = m.get(
                'tfidf_plausibility_floor', 0.12)
            self.rrf_k                    = m.get('rrf_k', 60)
            self.calibrator               = m.get('calibrator',
                                                   PlattCalibrator())
            self._sparse_dataset          = m.get('sparse_dataset', False)
            self.sbert_model_name         = m.get('sbert_model_name', None)
            self.sbert_model_type         = m.get('sbert_model_type', 'sbert')
            self._using_general_bert      = m.get('using_general_bert', False)
            self.id_to_index = {
                p: i for i, p in enumerate(self.patent_ids_ordered)}
            if FAISS_AVAILABLE and self.patent_embeddings is not None:
                self.sbert_index = self.build_faiss_index(
                    self.patent_embeddings.cpu().numpy().astype('float32'))
            logger.info(f"Cache loaded (model={cached_model})")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return False

    # ============================================================
    # NOVELTY PREDICTION — v9 decision hierarchy  (FIX-V9-4)
    # ============================================================

    def predict_novelty(self, new_text, top_k=50):
        """
        v9 Decision Hierarchy.

        Mode-aware thresholds are pulled from self._thr, which is set
        at construction time based on USE_PATENT_BERT.

        Patent-BERT mode  — tighter drift thresholds (BERT is trustworthy):
          Rule 1: top_tfidf < 0.12  AND  top_bert > 0.85  → ACCEPT
          Rule 2: median_top5_tfidf < 0.08  AND  max_cal >= floor  → ACCEPT
          Rule 3: max_tfidf < 0.10  AND  max_hybrid < 0.45  → ACCEPT

        General-BERT mode — wider drift thresholds (v8 values):
          Rule 1: top_tfidf < 0.25  AND  top_bert > 0.72  → ACCEPT
          Rule 2: median_top5_tfidf < 0.10  AND  max_cal >= floor  → ACCEPT
          Rule 3: max_tfidf < 0.15  AND  max_hybrid < 0.55  → ACCEPT

        Rules 4–6 are identical in both modes.
        """
        if self.patents_df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 70)
        print(f"NOVELTY PREDICTION  [{self._mode.upper()} mode]")
        print("=" * 70)
        print(f"\n[Query]: {new_text.strip()[:150]}...")

        t0 = time.time()
        results = self.compute_hybrid_similarity(new_text, top_k=top_k)
        elapsed = time.time() - t0

        if not results:
            return "[ERROR] No results", []

        cal_scores    = [r['calibrated_score'] for r in results]
        tfidf_scores  = [r['tfidf_sim']         for r in results]
        bert_scores   = [r['bert_sim']           for r in results]
        hybrid_scores = [r['hybrid_sim']         for r in results]

        max_cal    = float(max(cal_scores))
        top3_avg   = float(np.mean(cal_scores[:3]))
        top5_avg   = float(np.mean(cal_scores[:5]))
        top_tfidf  = float(tfidf_scores[0])
        top_bert   = float(bert_scores[0])
        max_tfidf  = float(max(tfidf_scores))
        max_hybrid = float(max(hybrid_scores))

        # Pull mode-aware thresholds
        T = self._thr

        # ── Rule 1: Semantic drift safeguard ─────────────────────────
        drift_detected = (top_tfidf < T["DRIFT_TFIDF_MAX"] and
                          top_bert  > T["DRIFT_BERT_MIN"])

        # ── Rule 2: Domain coherence check ───────────────────────────
        top5_tfidf_vals   = tfidf_scores[:min(5, len(tfidf_scores))]
        median_top5_tfidf = float(np.median(top5_tfidf_vals))
        coherence_failed  = (median_top5_tfidf < T["COHERENCE_TFIDF_MEDIAN"] and
                             max_cal >= self.novelty_floor)

        # ── Rule 3: TF-IDF override ───────────────────────────────────
        tfidf_override = (max_tfidf < T["TFIDF_OVERRIDE_MAX"] and
                          max_hybrid < T["HYBRID_OVERRIDE_MAX"])

        # ── Rule 4: Out-of-domain floor ──────────────────────────────
        out_of_domain = max_cal < self.novelty_floor

        # ── Rule 5: Dual-condition rejection ─────────────────────────
        cond_max    = max_cal  >= self.final_threshold
        cond_top3   = top3_avg >= (self.final_threshold - 0.10)
        dual_reject = cond_max and cond_top3

        # ── Apply hierarchy ───────────────────────────────────────────
        if drift_detected:
            reason     = (f"BERT Semantic Drift — top TF-IDF ({top_tfidf:.3f}) "
                          f"< {T['DRIFT_TFIDF_MAX']} with high BERT "
                          f"({top_bert:.3f}) > {T['DRIFT_BERT_MIN']}")
            is_novel   = True
            confidence = "High (drift safeguard triggered)"

        elif coherence_failed:
            reason     = (f"Domain Incoherence — median top-5 TF-IDF "
                          f"({median_top5_tfidf:.3f}) < "
                          f"{T['COHERENCE_TFIDF_MEDIAN']}; retrieved results "
                          f"do not share the query domain")
            is_novel   = True
            confidence = "High (domain coherence check triggered)"

        elif tfidf_override:
            reason     = "TF-IDF Override — no keyword overlap in any result"
            is_novel   = True
            confidence = "High (TF-IDF override triggered)"

        elif out_of_domain:
            reason     = "Out-of-domain — max calibrated score below novelty floor"
            is_novel   = True
            confidence = "High (out-of-domain)"

        elif dual_reject:
            gap        = abs(max_cal - self.final_threshold)
            reason     = "Strong prior art match found"
            is_novel   = False
            confidence = ("High"   if gap > 0.20 else
                          "Medium" if gap > 0.08 else
                          "Low — manual review recommended")
        else:
            reason     = "Below rejection threshold"
            is_novel   = True
            confidence = ("Medium" if abs(max_cal - self.final_threshold) > 0.08
                          else "Low — manual review recommended")

        decision = ("[ACCEPT] Potentially Novel" if is_novel
                    else "[REJECT] Not Novel — Prior Art Detected")

        # ── Print diagnostics ─────────────────────────────────────────
        print(f"\n[Scores]:")
        print(f"   Max Calibrated:           {max_cal:.4f}  ({max_cal*100:.1f}%)")
        print(f"   Top-3 Avg Calibrated:     {top3_avg:.4f}  ({top3_avg*100:.1f}%)")
        print(f"   Top-5 Avg Calibrated:     {top5_avg:.4f}  ({top5_avg*100:.1f}%)")
        print(f"   Top-result TF-IDF:        {top_tfidf:.4f}  "
              f"(drift if < {T['DRIFT_TFIDF_MAX']})")
        print(f"   Top-result BERT:          {top_bert:.4f}  "
              f"(drift if > {T['DRIFT_BERT_MIN']} with low TF-IDF)")
        print(f"   Median top-5 TF-IDF:      {median_top5_tfidf:.4f}  "
              f"(coherence floor: {T['COHERENCE_TFIDF_MEDIAN']})")
        print(f"   Max TF-IDF (all results): {max_tfidf:.4f}")
        print(f"   Max Hybrid (all results): {max_hybrid:.4f}")

        print(f"\n[Decision Rules — v9 {self._mode} mode]:")
        print(f"   Rule 1 — Drift Safeguard:  "
              f"{'⚡ TRIGGERED' if drift_detected else '○ skip'}"
              f"  [top_tfidf={top_tfidf:.3f}<{T['DRIFT_TFIDF_MAX']}"
              f" & top_bert={top_bert:.3f}>{T['DRIFT_BERT_MIN']}]")
        print(f"   Rule 2 — Domain Coherence: "
              f"{'⚡ TRIGGERED' if coherence_failed else '○ skip'}"
              f"  [median_top5={median_top5_tfidf:.3f}<{T['COHERENCE_TFIDF_MEDIAN']}"
              f" & max_cal={max_cal:.3f}>={self.novelty_floor:.3f}]")
        print(f"   Rule 3 — TF-IDF Override:  "
              f"{'⚡ TRIGGERED' if tfidf_override else '○ skip'}"
              f"  [max_tfidf={max_tfidf:.3f}<{T['TFIDF_OVERRIDE_MAX']}"
              f" & hybrid={max_hybrid:.3f}<{T['HYBRID_OVERRIDE_MAX']}]")
        print(f"   Rule 4 — Out-of-domain:    "
              f"{'⚡ TRIGGERED' if out_of_domain else '○ skip'}"
              f"  [max_cal={max_cal:.3f}<floor={self.novelty_floor:.3f}]")
        print(f"   Rule 5 — Dual Rejection:   "
              f"{'⚡ WOULD REJECT' if dual_reject else '○ no rejection'}"
              f"  [cond1={cond_max}: {max_cal:.3f}>={self.final_threshold:.3f},"
              f" cond2={cond_top3}: {top3_avg:.3f}>="
              f"{self.final_threshold-0.10:.3f}]")

        print(f"\n[Config — v9 {self._mode} mode]:")
        print(f"   USE_PATENT_BERT : {self.use_patent_bert}")
        print(f"   Threshold       : {self.final_threshold:.3f}")
        print(f"   Novelty floor   : {self.novelty_floor:.3f}")
        print(f"   Drift guard     : top_tfidf<{T['DRIFT_TFIDF_MAX']}"
              f" & top_bert>{T['DRIFT_BERT_MIN']}")
        print(f"   Coherence guard : median_top5_tfidf<{T['COHERENCE_TFIDF_MEDIAN']}")
        print(f"   TF-IDF override : max_tfidf<{T['TFIDF_OVERRIDE_MAX']}"
              f" & hybrid<{T['HYBRID_OVERRIDE_MAX']}")
        print(f"   Model           : {self.sbert_model_name}"
              + (" ✓ [patent]" if not self._using_general_bert
                 else " ⚠ [general]"))
        print(f"   Weights         : TF-IDF={self.w_tfidf:.3f},"
              f" BERT={self.w_bert:.3f}")
        print(f"   Inference       : {elapsed:.3f}s")
        print(f"   Reason          : {reason}")
        print(f"   Confidence      : {confidence}")

        print(f"\n{'='*60}")
        print(f"MOST SIMILAR PATENT")
        print(f"{'='*60}")
        top = results[0]
        dfl = ("⚠ LIKELY DRIFT"
               if (top['bert_sim'] > T["DRIFT_BERT_MIN"] and
                   top['tfidf_sim'] < T["DRIFT_TFIDF_MAX"]) else "")
        print(f"  ID        : {top['patent_id']}")
        print(f"  Title     : {top['title'][:80]}...")
        print(f"  TF-IDF    : {top['tfidf_sim']:.4f}  "
              f"{'⚠ LOW' if top['tfidf_sim'] < T['DRIFT_TFIDF_MAX'] else ''}")
        print(f"  BERT      : {top['bert_sim']:.4f}  {dfl}")
        print(f"  Hybrid    : {top['hybrid_sim']:.4f}")
        print(f"  Calibrated: {top['calibrated_score']:.4f}")

        print(f"\n{'='*60}")
        print(f"TOP {min(10, len(results))} SIMILAR PATENTS")
        print(f"{'='*60}")
        for i, r in enumerate(results[:10], 1):
            df_flag = (" ⚠" if (r['bert_sim'] > T["DRIFT_BERT_MIN"] and
                                  r['tfidf_sim'] < T["DRIFT_TFIDF_MAX"])
                       else "")
            print(f"\n{i:2d}. {r['patent_id']}  {r['title'][:60]}...")
            print(f"    TF={r['tfidf_sim']:.4f} | "
                  f"BERT={r['bert_sim']:.4f}{df_flag} | "
                  f"Hybrid={r['hybrid_sim']:.4f} | "
                  f"Cal={r['calibrated_score']:.4f}")

        print(f"\n{'='*60}")
        print(f"  [DECISION]:   {decision}")
        print(f"  [REASON]:     {reason}")
        print(f"  [CONFIDENCE]: {confidence}")
        print(f"{'='*60}")

        return decision, results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PATENT NOVELTY CHECK SYSTEM — CORRECTED VERSION v9")
    print("=" * 70)
    print(f"\nAuthor    : Devika Bakshi (122CS0301)")
    print(f"Supervisor: Asst. Prof. Sumanta Pyne")
    print(f"Institute : NIT Rourkela")
    print(f"Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n[USE_PATENT_BERT = {USE_PATENT_BERT}]")
    print(f"  {'→ Loading anferico/bert-for-patents (domain-adapted)' if USE_PATENT_BERT else '→ Loading BAAI/bge-base-en-v1.5 (general + v9 safeguards)'}")

    print("\n" + "─" * 70)
    print("SETUP NOTES")
    print("─" * 70)
    if USE_PATENT_BERT:
        print("Patent-BERT mode requires:")
        print("  pip install sentence-transformers protobuf sentencepiece")
        print("After switching USE_PATENT_BERT, cache is auto-invalidated.")
    else:
        print("General-BERT mode — no extra installs needed.")
        print("Drift safeguards compensate for embedding space collapse.")
    print("─" * 70)

    system = PatentNoveltySystem(use_patent_bert=USE_PATENT_BERT)

    # ── Dataset ─────────────────────────────────────────────────────
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

    # ── Change detection (includes model name in hash — FIX-V9-6) ───
    changed, current_hash, prev_hash = system.check_dataset_changed(patents_df)

    if changed:
        print(f"\n[INFO] Dataset / model changed — recomputing all embeddings")
        print(f"   Hash: {prev_hash[:8] if prev_hash else 'None'} "
              f"→ {current_hash[:8]}")
        system.compute_embeddings()
        system.train_hybrid_model(positive_pairs)
        system.save_dataset_hash(current_hash)
    else:
        print(f"\n[OK] Dataset + model unchanged ({current_hash[:8]})")
        if not system.load_cached_embeddings():
            print("   Cache invalid — recomputing...")
            system.compute_embeddings()
            system.train_hybrid_model(positive_pairs)
            system.save_dataset_hash(current_hash)
        else:
            system.init_sbert()

    # ── Test set ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET")
    print("=" * 70)
    n_test   = min(200, len(positive_pairs))
    test_pos = positive_pairs.sample(n=n_test, random_state=123).copy()
    test_pos['label'] = 1
    test_neg = system.get_mixed_negatives(
        test_pos, n_test, 0.65, np.random.RandomState(123))
    test_pairs = (pd.concat([test_pos, test_neg], ignore_index=True)
                  .sample(frac=1, random_state=123)
                  .reset_index(drop=True))
    print(f"Test: {len(test_pairs)} "
          f"(pos={test_pos.shape[0]}, neg={test_neg.shape[0]})")

    metrics = system.evaluate_model(test_pairs, eval_top_k=100)

    # ── Sample predictions ───────────────────────────────────────────
    #
    # Case A: Deep RL + CNN — should REJECT (clear prior art in DB)
    #   Patent-BERT:  top_tfidf ~0.34 → above 0.12 → Rule 1 skips → dual reject ✓
    #   General-BERT: top_tfidf ~0.34 → above 0.25 → Rule 1 skips → dual reject ✓
    #
    # Case B: RLHF / LLM — should ACCEPT (post-2022 technology)
    #   Patent-BERT:  true semantic similarity will be lower for unrelated patents
    #                 top_bert expected ~0.65-0.70 < 0.85 → Rule 1 skips but
    #                 calibrated score should be lower → floor or below threshold ✓
    #   General-BERT: top_tfidf ~0.22 < 0.25 + top_bert ~0.82 > 0.72 → drift ✓
    #
    # Case C: Bicycle lock — should ACCEPT (completely unrelated domain)
    #   Patent-BERT:  top_bert expected ~0.55-0.65 for unrelated; calibrated low ✓
    #   General-BERT: top_tfidf ~0.12 < 0.25 → drift Rule 1 triggers ✓

    test_cases = {
        "A — Deep RL + CNN (EXPECT: REJECT)": """
        A neural network processing system and method for real-time pattern recognition
        using deep reinforcement learning and convolutional neural network layers.
        The system employs adaptive learning rates and backpropagation to train an
        ensemble of neural networks for object detection and image classification,
        achieving superior performance over prior art neural network methods.
        """,

        "B — RLHF / LLM alignment (EXPECT: ACCEPT — post-2022 technology)": """
        A method for aligning large language models using reinforcement learning from
        human feedback (RLHF), comprising: a reward model fine-tuned on pairwise
        human preference annotations; a policy model trained with proximal policy
        optimization (PPO) and KL-divergence regularisation against a frozen reference
        model; and a constitutional AI self-critique loop that iteratively refines
        outputs to reduce harmful, toxic, and deceptive content.
        """,

        "C — Bicycle combination lock (EXPECT: ACCEPT — unrelated domain)": """
        A portable bicycle security device comprising a hardened steel shackle
        and a four-digit numeric combination dial mechanism. The user selects a
        custom numeric code by rotating numbered discs to align in sequence,
        releasing the shackle. Housing is weather-sealed with rubber gaskets.
        No electronic components. Purely mechanical design.
        """,
    }

    print("\n" + "=" * 70)
    print(f"SAMPLE NOVELTY PREDICTIONS — v9 [{system._mode.upper()} MODE]")
    print("=" * 70)

    decisions = {}
    for label, text in test_cases.items():
        print(f"\n{'─'*70}")
        print(f"  Case {label}")
        print(f"{'─'*70}")
        d, _ = system.predict_novelty(text, top_k=50)
        decisions[label] = d

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — v9  [{system._mode.upper()} MODE]")
    print(f"{'='*70}")
    print(f"  USE_PATENT_BERT   : {system.use_patent_bert}")
    print(f"  BERT Model        : {system.sbert_model_name}")
    if system._using_general_bert:
        print(f"  ⚠  General-purpose model — v9 drift safeguards (general mode)")
        print(f"     → pip install protobuf sentencepiece  to enable patent BERT")
    else:
        print(f"  ✓  Patent-specific model — tighter drift thresholds active")
    print(f"  Threshold         : {system.final_threshold:.3f}")
    print(f"  Novelty floor     : {system.novelty_floor:.3f}")
    T = system._thr
    print(f"\n  v9 Decision Rules [{system._mode} mode]:")
    print(f"    Rule 1 Drift:     top_tfidf<{T['DRIFT_TFIDF_MAX']}"
          f" & top_bert>{T['DRIFT_BERT_MIN']} → ACCEPT")
    print(f"    Rule 2 Coherence: median_top5_tfidf<{T['COHERENCE_TFIDF_MEDIAN']}"
          f" → ACCEPT")
    print(f"    Rule 3 Override:  max_tfidf<{T['TFIDF_OVERRIDE_MAX']}"
          f" & hybrid<{T['HYBRID_OVERRIDE_MAX']} → ACCEPT")
    print(f"    Rule 4 Floor:     max_cal<{system.novelty_floor:.3f} → ACCEPT")
    print(f"    Rule 5 Reject:    max_cal>={system.final_threshold:.3f}"
          f" & top3>={system.final_threshold-0.10:.3f} → REJECT")
    print(f"  Fusion weights    : TF-IDF={system.w_tfidf:.3f},"
          f" BERT={system.w_bert:.3f}")
    print(f"  Calibrator fitted : {system.calibrator.fitted}")

    if metrics:
        print(f"\n  Test Results:")
        print(f"    Accuracy    : {metrics['accuracy']:.3f}")
        print(f"    Bal. Acc.   : {metrics['balanced_accuracy']:.3f}")
        print(f"    Precision   : {metrics['precision']:.3f}")
        print(f"    Recall      : {metrics['recall']:.3f}")
        print(f"    F1          : {metrics['f1']:.3f}")
        if not np.isnan(metrics.get('auc_roc', float('nan'))):
            print(f"    AUC-ROC     : {metrics['auc_roc']:.3f}")
        for k in [10, 20, 50, 100]:
            r = metrics.get(f'recall_at_{k}', 0.0)
            print(f"    Recall@{k:<4}  : {r:.3f}  ({r*100:.1f}%)")

    print(f"\n  Sample Decisions:")
    for label, d in decisions.items():
        ok = (("REJECT" in label and "REJECT" in d) or
              ("ACCEPT" in label and "ACCEPT" in d))
        print(f"    {'✓' if ok else '✗'} {label[:55]}... → {d}")

    print(f"\n  v9 KEY ADDITIONS vs v8:")
    print(f"    FIX-V9-1: USE_PATENT_BERT flag — single switch for dual-mode")
    print(f"    FIX-V9-2: PatentBERTWrapper always L2-normalises embeddings")
    print(f"    FIX-V9-3: _encode_text() re-normalises as hard safety guarantee")
    print(f"    FIX-V9-4: Mode-aware thresholds (patent vs general sets)")
    print(f"    FIX-V9-5: Cache versioning — model name stored + verified on load")
    print(f"    FIX-V9-6: Cache hash includes model name → auto-invalidated on switch")
    print(f"    FIX-V9-7: Fusion weight bounds widen in patent-BERT mode")
    print(f"    FIX-V9-8: Sparsity floor 0.45 → 0.48 in patent-BERT mode")
    print(f"    FIX-V9-9: Embedding norm verification printed after compute")
    print(f"\n  System ready.")


if __name__ == "__main__":
    main()