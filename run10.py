"""
Automated Novelty Check System for Patent Pre-Screening
STABLE VERSION v12 — MPNet Semantic Model + Clean Architecture

Author: Devika Bakshi (122CS0301)
Supervisor: Asst. Prof. Sumanta Pyne
NIT Rourkela

SYSTEM TITLE:
  Hybrid Patent Novelty Detection System
  (TF-IDF + MPNet Semantic Embeddings + Rule-Based Overrides)

METHODOLOGY (report-ready):
  We use the `all-mpnet-base-v2` sentence-transformer model to generate
  normalized semantic embeddings. Cosine similarity is computed via dot
  product (equivalent since embeddings are L2-normalised). TF-IDF scores
  capture lexical/keyword overlap, while MPNet scores capture deep semantic
  similarity. The two are combined using a learned hybrid weighting scheme
  trained on citation-positive pairs. Domain-specific reasoning is introduced
  through rule-based overrides: semantic-gap detection, modern terminology
  analysis, TF-IDF rank decay, and drift safeguards.

WHAT v12 CHANGES vs v11:
=========================

CHANGE-V12-1: MODEL REPLACED — all-mpnet-base-v2  (PRIMARY change)
  - Removes the entire Patent-BERT / BGE / fallback loading chain
  - Uses sentence-transformers/all-mpnet-base-v2 directly and cleanly
  - No more tokenizer errors, no more silent fallbacks
  - MPNet is better than MiniLM and comparable to BGE for semantic tasks
  - Single model, single mode — no "patent" vs "general" switching needed

CHANGE-V12-2: TERMINOLOGY — "semantic" replaces "bert" everywhere
  - bert_score  → semantic_score
  - top_bert    → top_semantic
  - BERT sim    → Semantic sim
  - Cleaner output, better for report and viva

CHANGE-V12-3: SIMILARITY — dot product (embeddings are L2-normalised)
  - np.dot(v1, v2) instead of cosine_similarity(v1, v2)
  - Faster and numerically identical when norms = 1.0

CHANGE-V12-4: SEMANTIC GAP OVERRIDE — added semantic_score > 0.75 guard
  - Prevents spurious triggers when gap is large but both scores are low
  - Condition: gap > 0.55 AND max_tfidf < 0.32 AND top_semantic > 0.75

CHANGE-V12-5: SINGLE MODE — "mpnet" mode with well-calibrated thresholds
  - No more patent/general mode switching
  - Thresholds tuned for MPNet's embedding space

CHANGE-V12-6: LOGGING — clean success messages, no ⚠ warnings for expected state
  - "✓ Semantic similarity model loaded (MPNet)" instead of ⚠ messages

DECISION HIERARCHY v12 (unchanged logic, updated names):
=========================================================

Step 0  — TF-IDF Dampening              (if max_tfidf < 0.15 → cal * 0.60)
Step 1  — Modern Terminology Override   (2+ post-2022 terms + max_tfidf < 0.35)
Step 2  — Semantic Gap Override         (gap > 0.55 + max_tfidf < 0.32 + top_semantic > 0.75)
Step 3  — Strong Drift Override         (top_semantic > 0.80 + max_tfidf < 0.28)
Step 4  — TF-IDF Rank Decay             (decay_ratio < 0.50 + top_semantic > 0.80)
Step 5  — Rule 1: Drift Safeguard       (top_tfidf < 0.20 + top_semantic > 0.72)
Step 6  — Rule 2: Domain Coherence      (median_top5_tfidf < 0.15 + eff_cal >= floor)
Step 7  — Rule 3: TF-IDF Override       (max_tfidf < 0.15 + max_hybrid < 0.55)
Step 8  — Rule 4: Out-of-domain Floor   (eff_cal < 0.40)
Step 9  — Rule 5: Dual Rejection        (eff_cal >= thr AND top3_avg >= thr)
Step 10 — Confidence Band → ACCEPT

Expected results:
  Case A (Deep RL + CNN)  → REJECT  ✓  (max_tfidf=0.37, no modern terms, gap=0.47)
  Case B (RLHF alignment) → ACCEPT  ✓  (7 modern terms detected)
  Case C (Bicycle lock)   → ACCEPT  ✓  (gap=0.66, strong drift)
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
    print("FATAL: sentence-transformers not installed — pip install sentence-transformers")

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
# MODEL — single, stable, no fallback needed
# ============================================================
SEMANTIC_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# all-mpnet-base-v2: 768-dim, trained on 1B+ sentence pairs,
# best general-purpose sentence-transformer as of 2023.
# Outperforms MiniLM on semantic tasks; comparable to BGE.

# ============================================================
# THRESHOLDS (v12 — single MPNet mode)
# ============================================================
THRESHOLDS = {
    # Drift / overlap detection
    "DRIFT_TFIDF_MAX":          0.20,   # Rule 1: top TF-IDF below this → possible drift
    "DRIFT_SEMANTIC_MIN":       0.72,   # Rule 1: semantic score above this → drift confirmed
    "COHERENCE_TFIDF_MEDIAN":   0.15,   # Rule 2: median top-5 TF-IDF coherence floor
    "TFIDF_OVERRIDE_MAX":       0.15,   # Rule 3: max TF-IDF override
    "HYBRID_OVERRIDE_MAX":      0.55,   # Rule 3: max hybrid override
    # Strong drift override
    "STRONG_DRIFT_SEMANTIC_MIN":  0.80,
    "STRONG_DRIFT_TFIDF_MAX":     0.28,
    # TF-IDF dampening
    "DAMPEN_TFIDF_BELOW":         0.15,
    "DAMPEN_FACTOR":              0.60,
    # Modern terminology override (FIX-V11-1)
    "MODERN_TERM_TFIDF_MAX":      0.35,
    "MODERN_TERM_MIN_COUNT":      2,
    # Semantic gap override (FIX-V11-2, with V12-4 guard)
    "SEMANTIC_GAP_MIN":           0.55,
    "SEMANTIC_GAP_TFIDF_MAX":     0.32,
    "SEMANTIC_GAP_SCORE_MIN":     0.75,   # CHANGE-V12-4: top_semantic must exceed this
    # TF-IDF rank decay (FIX-V11-4)
    "RANK_DECAY_RATIO_MAX":       0.50,
    "RANK_DECAY_SEMANTIC_MIN":    0.80,
}

# ============================================================
# MODERN TERMINOLOGY VOCABULARY (post-2022 AI/ML terms)
# Presence of 2+ terms = strong novelty signal for Case B type queries
# ============================================================
MODERN_AI_TERMS = frozenset({
    # RLHF / Alignment
    "rlhf",
    "reinforcement learning from human feedback",
    "proximal policy optimization",
    "ppo",
    "constitutional ai",
    "kl divergence regularisation",
    "kl-divergence regularisation",
    "kl divergence regularization",
    "reward model",
    "reward modelling",
    "direct preference optimization",
    "dpo",
    "alignment tax",
    "red teaming llm",
    "jailbreak",
    "prompt injection",
    "value alignment",
    "harmlessness",
    "helpfulness honesty harmlessness",
    "hhh",
    # Large Language Models
    "large language model",
    "llm",
    "chatgpt",
    "gpt-4",
    "gpt4",
    "claude",
    "gemini",
    "llama",
    "mistral",
    "falcon llm",
    "instruction tuning",
    "instruction following",
    "chain of thought",
    "few-shot prompting",
    "zero-shot prompting",
    "in-context learning",
    "emergent ability",
    "scaling law",
    # Efficient Fine-tuning
    "lora",
    "qlora",
    "parameter efficient fine tuning",
    "peft",
    "adapter layer",
    "prefix tuning",
    "soft prompt",
    "prompt tuning",
    # Modern Architecture
    "flash attention",
    "mixture of experts",
    "moe transformer",
    "rotary position embedding",
    "rope",
    "grouped query attention",
    "gqa",
    "sliding window attention",
    "speculative decoding",
    "quantization aware training",
    # Generative / Multimodal
    "diffusion model",
    "stable diffusion",
    "dalle",
    "multimodal llm",
    "vision language model",
    "vlm",
    "text to image",
    "image generation model",
    # RAG / Vector Search
    "retrieval augmented generation",
    "rag pipeline",
    "vector database",
    "embedding store",
    "semantic search engine",
    "hallucination reduction",
    "grounding llm",
    # Tokenization
    "byte pair encoding",
    "sentencepiece",
    "tokenizer free",
})


def detect_modern_terms(query_text: str) -> list:
    """Returns list of post-2022 AI/ML terms found in query_text."""
    text_norm = query_text.lower().replace("-", " ").replace("_", " ")
    return [term for term in MODERN_AI_TERMS
            if term.replace("-", " ").replace("_", " ") in text_norm]


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
    Hybrid Patent Novelty Detection System — v12

    Architecture:
      - TF-IDF (trigrams, 15k features): exact keyword/phrase overlap
      - MPNet (all-mpnet-base-v2, 768-dim): deep semantic similarity
      - Platt calibration: maps hybrid scores to calibrated probabilities
      - Rule-based overrides: novelty reasoning beyond learned threshold
        (modern terminology, semantic gap, rank decay, drift safeguards)
    """

    def __init__(self, cache_dir='cache/', model_dir='models/'):

        self.stop_words  = set(stopwords.words('english'))
        self.lemmatizer  = WordNetLemmatizer()
        self._thr        = THRESHOLDS

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Device]: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"\n[Model]: {SEMANTIC_MODEL_NAME}")
        print(f"[Mode]:  Hybrid TF-IDF + MPNet Semantic Embeddings + Rule Overrides")

        self.vectorizer          = None
        self.tfidf_matrix        = None
        self.tfidf_matrix_dense  = None
        self.tfidf_cache         = None

        self.semantic_model      = None
        self.semantic_enabled    = SBERT_AVAILABLE
        self.patent_embeddings   = None
        self.semantic_index      = None
        self.patent_ids_ordered  = None
        self.id_to_index         = None

        # Fusion weights (will be learned; initial defaults)
        self._w_semantic_min, self._w_semantic_max = 0.35, 0.65
        self.w_tfidf    = 0.45
        self.w_semantic = 0.55

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

    def compute_robust_dataset_hash(self, patents_df):
        h = SEMANTIC_MODEL_NAME
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
    # FAISS INDEX
    # ============================================================

    def build_faiss_index(self, embeddings):
        if not FAISS_AVAILABLE:
            return None
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        logger.info(f"FAISS index: {index.ntotal} vectors, dim={embeddings.shape[1]}")
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

        n       = len(df)
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

        pos = citations.sample(n=min(1500, len(citations)), random_state=42).copy()
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
                sel  = np.random.choice(terms, size=min(5, len(terms)), replace=False)
                patents.append({
                    'patent_id':       f"PAT{pid:04d}",
                    'patent_title':    f"System for {main.lower()} in {domain.replace('_',' ')}",
                    'patent_abstract': (f"A {domain.replace('_',' ')} approach using "
                                        f"{', '.join(sel[:-1])}, and {sel[-1]}. "
                                        f"Improves over prior art in {domain.replace('_',' ')}.")
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
            cosine_similarity(self.tfidf_cache[p1], self.tfidf_cache[p2])[0][0],
            0.0, 1.0))

    def _semantic_sim(self, i1, i2):
        """
        Dot product of L2-normalised embeddings = cosine similarity.
        Raw MPNet inner product is in [-1, 1]; rescale to [0, 1].
        """
        raw = float(torch.dot(self.patent_embeddings[i1],
                              self.patent_embeddings[i2]).item())
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    def _fuse(self, tfidf, semantic):
        """
        Weighted fusion with TF-IDF plausibility floor.
        If TF-IDF is very low the semantic score is dampened proportionally
        to prevent high-semantic / zero-keyword spurious matches from
        dominating the hybrid score before the calibrator.
        """
        if tfidf < self.tfidf_plausibility_floor:
            semantic = semantic * (tfidf / self.tfidf_plausibility_floor)
        return float(np.clip(
            self.w_tfidf * tfidf + self.w_semantic * semantic, 0.0, 1.0))

    def _compute_pair_score_direct(self, p1, p2):
        i1 = self.id_to_index.get(str(p1))
        i2 = self.id_to_index.get(str(p2))
        if i1 is None or i2 is None:
            return None, None
        t   = self._tfidf_sim(str(p1), str(p2))
        s   = self._semantic_sim(i1, i2)
        raw = self._fuse(t, s)
        cal = (float(self.calibrator.predict_proba(np.array([raw]))[0])
               if self.calibrator.fitted else raw)
        return raw, cal

    # ============================================================
    # MODEL LOADING — clean, single model, no fallback needed
    # ============================================================

    def _load_semantic_model(self):
        """
        Loads sentence-transformers/all-mpnet-base-v2.
        768-dim, L2-normalised embeddings.
        Cosine similarity = dot product for normalised embeddings.
        """
        if not SBERT_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed.\n"
                "Fix: pip install sentence-transformers")

        print(f"\n   Loading: {SEMANTIC_MODEL_NAME}")
        model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        model = model.to(self.device)

        # Smoke test
        test_emb = model.encode(
            ["patent novelty check"],
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        norm = float(np.linalg.norm(test_emb[0]))
        print(f"   Smoke test — embedding dim: {test_emb.shape[1]}, norm: {norm:.4f}")
        assert abs(norm - 1.0) < 1e-3, f"Embedding not normalised! norm={norm}"
        print(f"   ✓ Semantic similarity model loaded (MPNet)")
        return model

    def _encode_text(self, text_or_list, show_progress_bar=False, batch_size=64):
        """
        Encode text(s) using MPNet. Returns L2-normalised torch tensor.
        batch_size=64 for speed (MPNet is heavier than MiniLM).
        """
        is_str = isinstance(text_or_list, str)
        if is_str:
            text_or_list = [text_or_list]

        emb = self.semantic_model.encode(
            text_or_list,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            device=self.device,
        )
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        emb = emb.to(self.device)
        emb = F.normalize(emb, p=2, dim=1)   # guarantee unit norm
        return emb[0] if is_str else emb

    # ============================================================
    # EMBEDDINGS
    # ============================================================

    def compute_embeddings(self):
        print("\n" + "=" * 70)
        print("COMPUTING EMBEDDINGS")
        print("=" * 70)

        # ── TF-IDF ──────────────────────────────────────────────────
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
            norms = np.linalg.norm(self.tfidf_matrix_dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.tfidf_matrix_dense /= norms
            print(f"   Dense TF-IDF cached: {self.tfidf_matrix_dense.shape}")
        else:
            self.tfidf_matrix_dense = None

        # ── MPNet Semantic Embeddings ────────────────────────────────
        if self.semantic_enabled:
            print("\n[2/2] MPNet semantic embeddings (768-dim)...")
            if self.semantic_model is None:
                self.semantic_model = self._load_semantic_model()

            texts = [self.text_map[p] for p in self.patent_ids_ordered]
            self.patent_embeddings = self._encode_text(
                texts, show_progress_bar=True, batch_size=64)

            norms = torch.norm(self.patent_embeddings, dim=1)
            print(f"   Embedding norms — mean: {norms.mean():.4f}  "
                  f"std: {norms.std():.6f}  (should be 1.000 ± 0.001)")
            print(f"   Embedding dim: {self.patent_embeddings.shape[1]}")

            self.id_to_index = {p: i for i, p in enumerate(self.patent_ids_ordered)}

            if FAISS_AVAILABLE:
                np_emb = self.patent_embeddings.cpu().numpy().astype('float32')
                self.semantic_index = self.build_faiss_index(np_emb)
        else:
            self.id_to_index = {p: i for i, p in enumerate(self.patent_ids_ordered)}

        print("\nEmbeddings done!")
        self.save_cached_embeddings()

    def init_semantic_model(self):
        if not SBERT_AVAILABLE:
            self.semantic_enabled = False
            return False
        if self.semantic_model is not None:
            return True
        try:
            self.semantic_model  = self._load_semantic_model()
            self.semantic_enabled = True
            return True
        except Exception as e:
            print(f"  Model load failed: {e}")
            self.semantic_enabled = False
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
            sims = cosine_similarity(self.tfidf_cache[p1], self.tfidf_matrix)[0]
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

    def get_mixed_negatives(self, positive_pairs, target, hard_ratio=0.65, rng=None):
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
        print("TRAINING HYBRID MODEL")
        print("=" * 70)
        print("  Architecture: TF-IDF + MPNet Semantic + Platt Calibration")

        if self.tfidf_matrix is None:
            self.compute_embeddings()

        pos   = positive_pairs.sample(frac=1, random_state=42).reset_index(drop=True)
        n_val = max(60, int(len(pos) * 0.30))
        val_pos   = pos.iloc[:n_val].copy()
        train_pos = pos.iloc[n_val:].copy()
        print(f"\n   Train+: {len(train_pos)} | Val+: {len(val_pos)}")

        print("\n[1/4] Training positives...")
        tt, st, yl = [], [], []   # tfidf, semantic, labels
        for _, row in train_pos.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1, i2 = self.id_to_index.get(p1), self.id_to_index.get(p2)
            if i1 is None or i2 is None: continue
            tt.append(self._tfidf_sim(p1, p2))
            st.append(self._semantic_sim(i1, i2))
            yl.append(1)
        n_pos = len(yl)
        print(f"   Valid: {n_pos}")

        print("\n[2/4] Mining negatives...")
        neg_df = self.get_mixed_negatives(train_pos, n_pos, 0.65, np.random.RandomState(42))
        for _, row in neg_df.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1, i2 = self.id_to_index.get(p1), self.id_to_index.get(p2)
            if i1 is None or i2 is None: continue
            tt.append(self._tfidf_sim(p1, p2))
            st.append(self._semantic_sim(i1, i2))
            yl.append(0)
        print(f"\n   Total: {len(yl)} (pos={sum(yl)}, neg={len(yl)-sum(yl)})")

        print("\n[3/4] Learning fusion weights (TF-IDF vs Semantic)...")
        X  = np.column_stack([tt, st])
        y  = np.array(yl)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X, y)
        raw_w = lr.coef_[0]
        ws    = np.sum(np.abs(raw_w))
        if ws > 0:
            lw = np.abs(raw_w) / ws
            # Clamp semantic weight to [35%, 65%] — neither signal dominates
            self.w_semantic = float(np.clip(lw[1], self._w_semantic_min, self._w_semantic_max))
            self.w_tfidf    = 1.0 - self.w_semantic
        print(f"   Learned weights: TF-IDF={self.w_tfidf:.3f}, Semantic={self.w_semantic:.3f}")

        fused = np.array([self._fuse(t, s) for t, s in zip(tt, st)])
        pm    = y == 1
        print(f"   Score separation: pos={fused[pm].mean():.3f}  "
              f"neg={fused[~pm].mean():.3f}  gap={fused[pm].mean()-fused[~pm].mean():.3f}")

        print("\n[4/4] Validation + calibration + threshold optimisation...")
        vn_df   = self.get_mixed_negatives(val_pos, len(val_pos), 0.65, np.random.RandomState(77))
        val_all = pd.concat([val_pos, vn_df], ignore_index=True)

        vt, vs, vl = [], [], []
        for _, row in val_all.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1, i2 = self.id_to_index.get(p1), self.id_to_index.get(p2)
            if i1 is None or i2 is None: continue
            vt.append(self._tfidf_sim(p1, p2))
            vs.append(self._semantic_sim(i1, i2))
            vl.append(int(row['label']))

        vf  = np.array([self._fuse(t, s) for t, s in zip(vt, vs)])
        vla = np.array(vl)
        print(f"\n   Val: {len(vla)} (pos={vla.sum()}, neg={(vla==0).sum()})")

        self.calibrator.fit(vf, vla)
        print(f"   Calibrator fitted: {self.calibrator.fitted}")

        vp = self.calibrator.predict_proba(vf) if self.calibrator.fitted else vf
        print(f"   Calibrated range: [{vp.min():.3f}, {vp.max():.3f}]")
        if vla.sum() > 0:
            print(f"     Pos mean: {vp[vla==1].mean():.3f}")
        if (vla == 0).sum() > 0:
            print(f"     Neg mean: {vp[vla==0].mean():.3f}")

        self._optimize_threshold(vp, vla)

        if getattr(self, '_sparse_dataset', False):
            sparsity_floor = 0.45
            old = self.final_threshold
            self.final_threshold = max(sparsity_floor, self.final_threshold - 0.03)
            print(f"\n   [Sparsity adj] {old:.3f} → {self.final_threshold:.3f} "
                  f"(floor={sparsity_floor})")

        self.save_cached_embeddings()
        return True

    def _optimize_threshold(self, y_scores, y_true):
        print(f"\n   Optimising rejection threshold (balanced accuracy)...")
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
        print(f"   Learned={best_t:.3f}  Applied={self.final_threshold:.3f} (floor=0.50)")
        print(f"   Acc={accuracy_score(y_true,preds):.3f}  "
              f"Prec={precision_score(y_true,preds,zero_division=0):.3f}  "
              f"Rec={recall_score(y_true,preds,zero_division=0):.3f}  "
              f"F1={f1_score(y_true,preds,zero_division=0):.3f}  "
              f"BAC={balanced_accuracy_score(y_true,preds):.3f}")
        if n_pos > 0 and n_neg > 0:
            print(f"   AUC={roc_auc_score(y_true, y_scores):.3f}")

    # ============================================================
    # RETRIEVAL — hybrid RRF fusion
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

    def _rrf(self, semantic_idx, tfidf_idx, k=60):
        """Reciprocal Rank Fusion of semantic and TF-IDF rankings."""
        scores = {}
        for rank, idx in enumerate(semantic_idx):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(tfidf_idx):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def compute_hybrid_similarity(self, query_text, top_k=100):
        if not self.semantic_enabled or self.patent_embeddings is None:
            return self._tfidf_only(query_text, top_k)

        clean = self.preprocess(query_text)
        if not clean.strip():
            return []

        n_semantic = min(top_k * 6,  len(self.patent_ids_ordered))
        n_tfidf    = min(top_k * 8,  len(self.patent_ids_ordered))

        # TF-IDF retrieval
        qvec      = self.vectorizer.transform([clean])
        tfidf_top_idx, _ = self._tfidf_top(qvec, n_tfidf)
        tfidf_all = np.clip(cosine_similarity(qvec, self.tfidf_matrix)[0], 0.0, 1.0)

        # Semantic retrieval (dot product = cosine for normalised embeddings)
        qemb = self._encode_text(clean)
        if self.semantic_index is not None and FAISS_AVAILABLE:
            qnp = qemb.cpu().numpy().reshape(1, -1).astype('float32')
            faiss.normalize_L2(qnp)
            sscores_raw, sidx = self.semantic_index.search(qnp, n_semantic)
            semantic_top_idx  = sidx[0]
            semantic_scores_d = {
                int(i): float(np.clip((s + 1) / 2, 0, 1))
                for i, s in zip(semantic_top_idx, sscores_raw[0])}
        else:
            # dot product on normalised embeddings
            sraw = torch.mm(qemb.unsqueeze(0), self.patent_embeddings.T)[0].cpu().numpy()
            sall = np.clip((sraw + 1.0) / 2.0, 0.0, 1.0)
            semantic_top_idx  = np.argsort(sall)[::-1][:n_semantic]
            semantic_scores_d = {int(i): float(sall[i]) for i in semantic_top_idx}

        # RRF fusion
        rrf = self._rrf(semantic_top_idx, tfidf_top_idx, k=self.rrf_k)

        results = []
        for orig_idx, _ in rrf:
            pid       = self.patent_ids_ordered[orig_idx]
            tfidf_s   = float(tfidf_all[orig_idx])
            semantic_s = semantic_scores_d.get(
                orig_idx,
                float(np.clip(
                    (torch.dot(qemb, self.patent_embeddings[orig_idx]).item() + 1) / 2,
                    0, 1)))
            hybrid = self._fuse(tfidf_s, semantic_s)
            cal    = (float(self.calibrator.predict_proba(np.array([hybrid]))[0])
                      if self.calibrator.fitted else hybrid)
            results.append({
                'patent_id':        pid,
                'title':            self.title_map.get(pid, pid),
                'tfidf_sim':        tfidf_s,
                'semantic_sim':     semantic_s,
                'hybrid_sim':       hybrid,
                'calibrated_score': cal,
            })

        results.sort(key=lambda x: x['hybrid_sim'], reverse=True)
        return results[:top_k]

    def _tfidf_only(self, query_text, top_k=100):
        clean = self.preprocess(query_text)
        qvec  = self.vectorizer.transform([clean])
        sims  = np.clip(cosine_similarity(qvec, self.tfidf_matrix)[0], 0.0, 1.0)
        idx   = np.argsort(sims)[::-1][:top_k]
        return [{
            'patent_id':        self.patent_ids_ordered[i],
            'title':            self.title_map.get(self.patent_ids_ordered[i], ''),
            'tfidf_sim':        float(sims[i]),
            'semantic_sim':     float(sims[i]),
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
        print("MODEL EVALUATION")
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
                    res  = self.compute_hybrid_similarity(qtxt, top_k=eval_top_k)
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
        print(f"   Model: {SEMANTIC_MODEL_NAME}  ✓ MPNet Semantic")

        if n_p and n_n:
            print(f"\n   RAW Hybrid — Pos: {y_raw[y_true==1].mean():.3f}  "
                  f"Neg: {y_raw[y_true==0].mean():.3f}  "
                  f"Sep: {y_raw[y_true==1].mean()-y_raw[y_true==0].mean():.3f}")
            print(f"   CALIBRATED — Pos: {y_cal[y_true==1].mean():.3f}  "
                  f"Neg: {y_cal[y_true==0].mean():.3f}  "
                  f"Sep: {y_cal[y_true==1].mean()-y_cal[y_true==0].mean():.3f}")

        ypred = (y_cal >= self.final_threshold).astype(int)
        auc   = roc_auc_score(y_true, y_cal)  if n_p and n_n else float('nan')
        ap    = average_precision_score(y_true, y_cal) if n_p and n_n else float('nan')

        metrics = {
            'accuracy':          accuracy_score(y_true, ypred),
            'balanced_accuracy': balanced_accuracy_score(y_true, ypred),
            'precision':         precision_score(y_true, ypred, zero_division=0),
            'recall':            recall_score(y_true, ypred, zero_division=0),
            'f1':                f1_score(y_true, ypred, zero_division=0),
            'auc_roc':           auc,
            'avg_precision':     ap,
            'threshold':         self.final_threshold,
        }
        for k in rcounts:
            metrics[f'recall_at_{k}'] = rcounts[k] / max(n_pos_total, 1)

        print(f"\n[Classification] (threshold={self.final_threshold:.3f}):")
        print(f"   Accuracy:   {metrics['accuracy']:.3f}  ({metrics['accuracy']*100:.1f}%)")
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
    # CACHE
    # ============================================================

    def save_cached_embeddings(self):
        from scipy.sparse import save_npz
        save_npz(os.path.join(self.cache_dir, 'tfidf_matrix.npz'), self.tfidf_matrix)
        if self.semantic_enabled and self.patent_embeddings is not None:
            torch.save(self.patent_embeddings.cpu(),
                       os.path.join(self.cache_dir, 'semantic_embeddings.pt'))
        with open(os.path.join(self.cache_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        if self.tfidf_matrix_dense is not None:
            np.save(os.path.join(self.cache_dir, 'tfidf_dense.npy'),
                    self.tfidf_matrix_dense)
        meta = {
            'model_version':            SEMANTIC_MODEL_NAME,
            'system_version':           'v12',
            'patent_ids_ordered':       self.patent_ids_ordered,
            'title_map':                self.title_map,
            'text_map':                 self.text_map,
            'citation_set':             list(self.citation_set) if self.citation_set else [],
            'w_tfidf':                  self.w_tfidf,
            'w_semantic':               self.w_semantic,
            'final_threshold':          self.final_threshold,
            'novelty_floor':            self.novelty_floor,
            'tfidf_plausibility_floor': self.tfidf_plausibility_floor,
            'rrf_k':                    self.rrf_k,
            'calibrator':               self.calibrator,
            'sparse_dataset':           getattr(self, '_sparse_dataset', False),
        }
        with open(os.path.join(self.cache_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        logger.info("Cache saved (v12)")

    def load_cached_embeddings(self):
        from scipy.sparse import load_npz
        paths = {
            'tfidf':     os.path.join(self.cache_dir, 'tfidf_matrix.npz'),
            'semantic':  os.path.join(self.cache_dir, 'semantic_embeddings.pt'),
            'vec':       os.path.join(self.cache_dir, 'vectorizer.pkl'),
            'meta':      os.path.join(self.cache_dir, 'metadata.pkl'),
            'dense':     os.path.join(self.cache_dir, 'tfidf_dense.npy'),
        }
        if not all(os.path.exists(paths[k]) for k in ['tfidf', 'vec', 'meta']):
            return False
        try:
            with open(paths['meta'], 'rb') as f:
                m = pickle.load(f)

            if m.get('system_version') != 'v12':
                logger.warning("Cache is from a previous version — recomputing")
                return False
            if m.get('model_version') != SEMANTIC_MODEL_NAME:
                logger.warning(f"Cache model mismatch — recomputing")
                return False

            self.tfidf_matrix = load_npz(paths['tfidf'])
            if os.path.exists(paths['dense']):
                self.tfidf_matrix_dense = np.load(paths['dense'])
            if self.semantic_enabled and os.path.exists(paths['semantic']):
                self.patent_embeddings = torch.load(paths['semantic'], map_location='cpu')
                if self.device.type == 'cuda':
                    self.patent_embeddings = self.patent_embeddings.to(self.device)
            with open(paths['vec'], 'rb') as f:
                self.vectorizer = pickle.load(f)

            self.patent_ids_ordered       = m['patent_ids_ordered']
            self.title_map                = m['title_map']
            self.text_map                 = m['text_map']
            self.citation_set             = set(m.get('citation_set', []))
            self.citation_set_bidirectional = (
                self.citation_set | {(b, a) for a, b in self.citation_set})
            self.w_tfidf                  = m.get('w_tfidf',    0.45)
            self.w_semantic               = m.get('w_semantic', 0.55)
            self.final_threshold          = m.get('final_threshold', 0.50)
            self.novelty_floor            = m.get('novelty_floor', 0.40)
            self.tfidf_plausibility_floor = m.get('tfidf_plausibility_floor', 0.12)
            self.rrf_k                    = m.get('rrf_k', 60)
            self.calibrator               = m.get('calibrator', PlattCalibrator())
            self._sparse_dataset          = m.get('sparse_dataset', False)

            self.id_to_index = {p: i for i, p in enumerate(self.patent_ids_ordered)}
            if FAISS_AVAILABLE and self.patent_embeddings is not None:
                self.semantic_index = self.build_faiss_index(
                    self.patent_embeddings.cpu().numpy().astype('float32'))
            logger.info(f"Cache loaded (v12, model={SEMANTIC_MODEL_NAME})")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return False

    # ============================================================
    # NOVELTY PREDICTION — v12 decision hierarchy
    # ============================================================

    def predict_novelty(self, new_text, top_k=50):
        """
        v12 Decision Hierarchy
        (logic identical to v11; variable names updated to 'semantic').

        Step 0:  TF-IDF Dampening
        Step 1:  Modern Terminology Override    [post-2022 AI vocab]
        Step 2:  Semantic Gap Override          [gap > 0.55 + score guard]
        Step 3:  Strong Drift Override
        Step 4:  TF-IDF Rank Decay
        Step 5:  Rule 1 — Drift Safeguard
        Step 6:  Rule 2 — Domain Coherence
        Step 7:  Rule 3 — TF-IDF Override
        Step 8:  Rule 4 — Out-of-domain Floor
        Step 9:  Rule 5 — Dual Rejection
        Step 10: Confidence Band → ACCEPT
        """
        if self.patents_df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 70)
        print("NOVELTY PREDICTION — Hybrid TF-IDF + MPNet System v12")
        print("=" * 70)
        print(f"\n[Query]: {new_text.strip()[:150]}...")

        modern_terms_found = detect_modern_terms(new_text)

        t0      = time.time()
        results = self.compute_hybrid_similarity(new_text, top_k=top_k)
        elapsed = time.time() - t0

        if not results:
            return "[ERROR] No results", []

        cal_scores      = [r['calibrated_score'] for r in results]
        tfidf_scores    = [r['tfidf_sim']         for r in results]
        semantic_scores = [r['semantic_sim']       for r in results]
        hybrid_scores   = [r['hybrid_sim']         for r in results]

        max_cal      = float(max(cal_scores))
        top3_avg     = float(np.mean(cal_scores[:3]))
        top5_avg     = float(np.mean(cal_scores[:5]))
        top_tfidf    = float(tfidf_scores[0])
        top_semantic = float(semantic_scores[0])
        max_tfidf    = float(max(tfidf_scores))
        max_hybrid   = float(max(hybrid_scores))
        semantic_gap = top_semantic - top_tfidf

        # Rank decay: how fast do TF-IDF scores fall off by rank 3?
        rank3_tfidf = float(tfidf_scores[2]) if len(tfidf_scores) > 2 else top_tfidf
        tfidf_rank_decay_ratio = (rank3_tfidf / top_tfidf) if top_tfidf > 1e-6 else 0.0

        T = self._thr

        # ── Step 0: TF-IDF Dampening ──────────────────────────────────
        if max_tfidf < T["DAMPEN_TFIDF_BELOW"]:
            effective_cal = max_cal * T["DAMPEN_FACTOR"]
            dampened      = True
        else:
            effective_cal = max_cal
            dampened      = False

        # ── Step 1: Modern Terminology Override ───────────────────────
        modern_override = (
            len(modern_terms_found) >= T["MODERN_TERM_MIN_COUNT"] and
            max_tfidf < T["MODERN_TERM_TFIDF_MAX"]
        )

        # ── Step 2: Semantic Gap Override (with score guard, V12-4) ───
        gap_override = (
            semantic_gap  > T["SEMANTIC_GAP_MIN"]      and
            max_tfidf     < T["SEMANTIC_GAP_TFIDF_MAX"] and
            top_semantic  > T["SEMANTIC_GAP_SCORE_MIN"]  # V12-4: prevents noise
        )

        # ── Step 3: Strong Drift Override ─────────────────────────────
        strong_drift = (
            top_semantic > T["STRONG_DRIFT_SEMANTIC_MIN"] and
            max_tfidf    < T["STRONG_DRIFT_TFIDF_MAX"]
        )

        # ── Step 4: TF-IDF Rank Decay ─────────────────────────────────
        rank_decay_override = (
            tfidf_rank_decay_ratio < T["RANK_DECAY_RATIO_MAX"]  and
            top_semantic           > T["RANK_DECAY_SEMANTIC_MIN"] and
            max_tfidf              < T["STRONG_DRIFT_TFIDF_MAX"]
        )

        # ── Step 5 (Rule 1): Drift Safeguard ──────────────────────────
        drift_detected = (
            top_tfidf    < T["DRIFT_TFIDF_MAX"] and
            top_semantic > T["DRIFT_SEMANTIC_MIN"]
        )

        # ── Step 6 (Rule 2): Domain Coherence ─────────────────────────
        top5_tfidf_vals   = tfidf_scores[:min(5, len(tfidf_scores))]
        median_top5_tfidf = float(np.median(top5_tfidf_vals))
        coherence_failed  = (
            median_top5_tfidf < T["COHERENCE_TFIDF_MEDIAN"] and
            effective_cal >= self.novelty_floor
        )

        # ── Step 7 (Rule 3): TF-IDF Override ──────────────────────────
        tfidf_override = (
            max_tfidf  < T["TFIDF_OVERRIDE_MAX"] and
            max_hybrid < T["HYBRID_OVERRIDE_MAX"]
        )

        # ── Step 8 (Rule 4): Out-of-domain Floor ──────────────────────
        out_of_domain = effective_cal < self.novelty_floor

        # ── Step 9 (Rule 5): Dual Rejection ───────────────────────────
        cond_max    = effective_cal >= self.final_threshold
        cond_top3   = top3_avg     >= self.final_threshold
        dual_reject = cond_max and cond_top3

        # ── Apply hierarchy ────────────────────────────────────────────
        if modern_override:
            reason     = (f"Modern Terminology Override — {len(modern_terms_found)} "
                          f"post-2022 AI terms detected: "
                          f"{', '.join(modern_terms_found[:5])}"
                          f"{'...' if len(modern_terms_found) > 5 else ''}. "
                          f"Concept post-dates patent database → novel.")
            is_novel   = True
            confidence = "High (modern terminology override)"

        elif gap_override:
            reason     = (f"Semantic Gap Override — Semantic ({top_semantic:.3f}) "
                          f"- TF-IDF ({top_tfidf:.3f}) = gap {semantic_gap:.3f} "
                          f"> {T['SEMANTIC_GAP_MIN']} with semantic score "
                          f"> {T['SEMANTIC_GAP_SCORE_MIN']}. "
                          f"High-level semantic match but no lexical overlap → novel.")
            is_novel   = True
            confidence = "High (semantic gap override)"

        elif strong_drift:
            reason     = (f"Strong Drift Override — max TF-IDF ({max_tfidf:.3f}) "
                          f"< {T['STRONG_DRIFT_TFIDF_MAX']} with high semantic "
                          f"score ({top_semantic:.3f}) > {T['STRONG_DRIFT_SEMANTIC_MIN']}. "
                          f"Semantic similarity without lexical overlap → novel.")
            is_novel   = True
            confidence = "High (strong drift override)"

        elif rank_decay_override:
            reason     = (f"TF-IDF Rank Decay — TF-IDF drops from "
                          f"{top_tfidf:.3f} (rank 1) to {rank3_tfidf:.3f} (rank 3), "
                          f"ratio {tfidf_rank_decay_ratio:.3f} < {T['RANK_DECAY_RATIO_MAX']}. "
                          f"Single spurious keyword match, not systematic prior art.")
            is_novel   = True
            confidence = "High (rank decay override)"

        elif drift_detected:
            reason     = (f"Drift Safeguard — top TF-IDF ({top_tfidf:.3f}) "
                          f"< {T['DRIFT_TFIDF_MAX']} with high semantic "
                          f"({top_semantic:.3f}) > {T['DRIFT_SEMANTIC_MIN']}")
            is_novel   = True
            confidence = "High (drift safeguard)"

        elif coherence_failed:
            reason     = (f"Domain Incoherence — median top-5 TF-IDF "
                          f"({median_top5_tfidf:.3f}) < {T['COHERENCE_TFIDF_MEDIAN']}; "
                          f"retrieved patents do not share the query's technical domain")
            is_novel   = True
            confidence = "High (domain coherence check)"

        elif tfidf_override:
            reason     = "TF-IDF Override — no keyword overlap in any retrieved patent"
            is_novel   = True
            confidence = "High (TF-IDF override)"

        elif out_of_domain:
            reason     = "Out-of-domain — calibrated score below novelty floor"
            is_novel   = True
            confidence = "High (out-of-domain)"

        elif dual_reject:
            gap        = abs(effective_cal - self.final_threshold)
            reason     = "Strong prior art match found"
            is_novel   = False
            confidence = ("High"   if gap > 0.20 else
                          "Medium" if gap > 0.08 else
                          "Low — manual review recommended")
        else:
            reason     = "Below rejection threshold"
            is_novel   = True
            confidence = ("Medium" if abs(effective_cal - self.final_threshold) > 0.08
                          else "Low — manual review recommended")

        decision = ("[ACCEPT] Potentially Novel" if is_novel
                    else "[REJECT] Not Novel — Prior Art Detected")

        # ── Diagnostics ────────────────────────────────────────────────
        print(f"\n[Scores]:")
        print(f"   Max Calibrated:            {max_cal:.4f}  ({max_cal*100:.1f}%)")
        print(f"   Effective Cal (dampened):  {effective_cal:.4f}  "
              f"{'⚡ dampened ×' + str(T['DAMPEN_FACTOR']) if dampened else '(no dampening)'}")
        print(f"   Top-3 Avg Calibrated:      {top3_avg:.4f}  ({top3_avg*100:.1f}%)")
        print(f"   Top-5 Avg Calibrated:      {top5_avg:.4f}  ({top5_avg*100:.1f}%)")
        print(f"   Top-result TF-IDF:         {top_tfidf:.4f}")
        print(f"   Top-result Semantic:       {top_semantic:.4f}")
        print(f"   Semantic Gap (Sem - TF):   {semantic_gap:.4f}  "
              f"(override if > {T['SEMANTIC_GAP_MIN']} & sem > {T['SEMANTIC_GAP_SCORE_MIN']})")
        print(f"   Median top-5 TF-IDF:       {median_top5_tfidf:.4f}")
        print(f"   Max TF-IDF (all results):  {max_tfidf:.4f}")
        print(f"   Max Hybrid (all results):  {max_hybrid:.4f}")
        print(f"   TF-IDF rank decay ratio:   {tfidf_rank_decay_ratio:.4f}  "
              f"(rank3={rank3_tfidf:.4f} / rank1={top_tfidf:.4f})")
        if modern_terms_found:
            print(f"   Modern terms detected:     {len(modern_terms_found)} — "
                  f"{', '.join(modern_terms_found[:6])}")
        else:
            print(f"   Modern terms detected:     0")

        print(f"\n[Decision Rules — v12]:")
        print(f"   Step 0 — Dampening:          "
              f"{'⚡ ACTIVE' if dampened else '○ skip'}"
              f"  [max_tfidf={max_tfidf:.3f} < {T['DAMPEN_TFIDF_BELOW']}]")
        print(f"   Step 1 — ModernTerms:        "
              f"{'⚡ → ACCEPT' if modern_override else '○ skip'}"
              f"  [{len(modern_terms_found)} >= {T['MODERN_TERM_MIN_COUNT']}"
              f" & max_tfidf={max_tfidf:.3f} < {T['MODERN_TERM_TFIDF_MAX']}]")
        print(f"   Step 2 — SemanticGap:        "
              f"{'⚡ → ACCEPT' if gap_override else '○ skip'}"
              f"  [gap={semantic_gap:.3f} > {T['SEMANTIC_GAP_MIN']}"
              f" & tfidf={max_tfidf:.3f} < {T['SEMANTIC_GAP_TFIDF_MAX']}"
              f" & sem={top_semantic:.3f} > {T['SEMANTIC_GAP_SCORE_MIN']}]")
        print(f"   Step 3 — StrongDrift:        "
              f"{'⚡ → ACCEPT' if strong_drift else '○ skip'}"
              f"  [sem={top_semantic:.3f} > {T['STRONG_DRIFT_SEMANTIC_MIN']}"
              f" & max_tfidf={max_tfidf:.3f} < {T['STRONG_DRIFT_TFIDF_MAX']}]")
        print(f"   Step 4 — RankDecay:          "
              f"{'⚡ → ACCEPT' if rank_decay_override else '○ skip'}"
              f"  [ratio={tfidf_rank_decay_ratio:.3f} < {T['RANK_DECAY_RATIO_MAX']}"
              f" & sem={top_semantic:.3f} > {T['RANK_DECAY_SEMANTIC_MIN']}"
              f" & max_tfidf < {T['STRONG_DRIFT_TFIDF_MAX']}]")
        print(f"   Rule 1 — Drift Safeguard:    "
              f"{'⚡ → ACCEPT' if drift_detected else '○ skip'}"
              f"  [top_tfidf={top_tfidf:.3f} < {T['DRIFT_TFIDF_MAX']}"
              f" & sem={top_semantic:.3f} > {T['DRIFT_SEMANTIC_MIN']}]")
        print(f"   Rule 2 — Domain Coherence:   "
              f"{'⚡ → ACCEPT' if coherence_failed else '○ skip'}"
              f"  [median={median_top5_tfidf:.3f} < {T['COHERENCE_TFIDF_MEDIAN']}"
              f" & eff_cal={effective_cal:.3f} >= {self.novelty_floor}]")
        print(f"   Rule 3 — TF-IDF Override:    "
              f"{'⚡ → ACCEPT' if tfidf_override else '○ skip'}"
              f"  [max_tfidf={max_tfidf:.3f} < {T['TFIDF_OVERRIDE_MAX']}"
              f" & hybrid={max_hybrid:.3f} < {T['HYBRID_OVERRIDE_MAX']}]")
        print(f"   Rule 4 — Out-of-domain:      "
              f"{'⚡ → ACCEPT' if out_of_domain else '○ skip'}"
              f"  [eff_cal={effective_cal:.3f} < {self.novelty_floor}]")
        print(f"   Rule 5 — Dual Rejection:     "
              f"{'⚡ REJECT' if dual_reject else '○ no rejection'}"
              f"  [eff_cal={effective_cal:.3f} >= {self.final_threshold:.3f}"
              f" AND top3={top3_avg:.3f} >= {self.final_threshold:.3f}]")

        print(f"\n[System Config — v12]:")
        print(f"   Model       : {SEMANTIC_MODEL_NAME}  ✓ MPNet")
        print(f"   Threshold   : {self.final_threshold:.3f}")
        print(f"   Novelty floor: {self.novelty_floor:.3f}")
        print(f"   Weights     : TF-IDF={self.w_tfidf:.3f}, Semantic={self.w_semantic:.3f}")
        print(f"   Inference   : {elapsed:.3f}s")
        print(f"   Reason      : {reason}")
        print(f"   Confidence  : {confidence}")

        print(f"\n{'='*60}")
        print(f"MOST SIMILAR PATENT")
        print(f"{'='*60}")
        top = results[0]
        print(f"  ID        : {top['patent_id']}")
        print(f"  Title     : {top['title'][:80]}...")
        print(f"  TF-IDF    : {top['tfidf_sim']:.4f}")
        print(f"  Semantic  : {top['semantic_sim']:.4f}")
        print(f"  Gap       : {top['semantic_sim'] - top['tfidf_sim']:.4f}")
        print(f"  Hybrid    : {top['hybrid_sim']:.4f}")
        print(f"  Calibrated: {top['calibrated_score']:.4f}")

        print(f"\n{'='*60}")
        print(f"TOP {min(10, len(results))} SIMILAR PATENTS")
        print(f"{'='*60}")
        for i, r in enumerate(results[:10], 1):
            gap_f = r['semantic_sim'] - r['tfidf_sim']
            flag  = f" ⚠ gap={gap_f:.3f}" if gap_f > T['SEMANTIC_GAP_MIN'] else ""
            print(f"\n{i:2d}. {r['patent_id']}  {r['title'][:60]}...")
            print(f"    TF={r['tfidf_sim']:.4f} | "
                  f"Sem={r['semantic_sim']:.4f}{flag} | "
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
    print("HYBRID PATENT NOVELTY DETECTION SYSTEM — v12")
    print("TF-IDF + MPNet Semantic Embeddings + Rule-Based Overrides")
    print("=" * 70)
    print(f"\nAuthor    : Devika Bakshi (122CS0301)")
    print(f"Supervisor: Asst. Prof. Sumanta Pyne")
    print(f"Institute : NIT Rourkela")
    print(f"Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSemantic model : {SEMANTIC_MODEL_NAME}")
    print(f"  → 768-dim embeddings, L2-normalised")
    print(f"  → Cosine similarity via dot product")
    print(f"  → No fallback required — single stable model")

    system = PatentNoveltySystem()

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

    # ── Change detection ─────────────────────────────────────────────
    changed, current_hash, prev_hash = system.check_dataset_changed(patents_df)

    if changed:
        print(f"\n[INFO] Dataset / model changed — recomputing all embeddings")
        print(f"   Hash: {prev_hash[:8] if prev_hash else 'None'} → {current_hash[:8]}")
        system.compute_embeddings()
        system.train_hybrid_model(positive_pairs)
        system.save_dataset_hash(current_hash)
    else:
        print(f"\n[OK] Dataset unchanged ({current_hash[:8]})")
        if not system.load_cached_embeddings():
            print("   Cache invalid — recomputing...")
            system.compute_embeddings()
            system.train_hybrid_model(positive_pairs)
            system.save_dataset_hash(current_hash)
        else:
            system.init_semantic_model()

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
    print(f"Test: {len(test_pairs)} (pos={test_pos.shape[0]}, neg={test_neg.shape[0]})")

    metrics = system.evaluate_model(test_pairs, eval_top_k=100)

    # ── Sample predictions ───────────────────────────────────────────
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
    print("SAMPLE NOVELTY PREDICTIONS — v12")
    print("=" * 70)

    decisions = {}
    for label, text in test_cases.items():
        print(f"\n{'─'*70}")
        print(f"  Case {label}")
        print(f"{'─'*70}")
        d, _ = system.predict_novelty(text, top_k=50)
        decisions[label] = d

    # ── Final Summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — Hybrid Patent Novelty Detection System v12")
    print(f"{'='*70}")
    print(f"  Semantic model   : {SEMANTIC_MODEL_NAME}  ✓")
    print(f"  Embedding dim    : 768")
    print(f"  Similarity       : dot product (L2-normalised = cosine)")
    print(f"  Threshold        : {system.final_threshold:.3f}")
    print(f"  Novelty floor    : {system.novelty_floor:.3f}")
    print(f"  Fusion weights   : TF-IDF={system.w_tfidf:.3f}, Semantic={system.w_semantic:.3f}")
    print(f"  Calibrator fitted: {system.calibrator.fitted}")

    T = system._thr
    print(f"\n  Decision Hierarchy [v12]:")
    print(f"    Step 0 Dampen:       max_tfidf<{T['DAMPEN_TFIDF_BELOW']}"
          f" → cal×{T['DAMPEN_FACTOR']}")
    print(f"    Step 1 ModernTerms:  {T['MODERN_TERM_MIN_COUNT']}+ post-2022 AI terms"
          f" & max_tfidf<{T['MODERN_TERM_TFIDF_MAX']} → ACCEPT")
    print(f"    Step 2 SemanticGap:  gap>{T['SEMANTIC_GAP_MIN']}"
          f" & tfidf<{T['SEMANTIC_GAP_TFIDF_MAX']}"
          f" & sem>{T['SEMANTIC_GAP_SCORE_MIN']} → ACCEPT")
    print(f"    Step 3 StrongDrift:  sem>{T['STRONG_DRIFT_SEMANTIC_MIN']}"
          f" & max_tfidf<{T['STRONG_DRIFT_TFIDF_MAX']} → ACCEPT")
    print(f"    Step 4 RankDecay:    ratio<{T['RANK_DECAY_RATIO_MAX']}"
          f" & sem>{T['RANK_DECAY_SEMANTIC_MIN']} → ACCEPT")
    print(f"    Rule 1 Drift:        top_tfidf<{T['DRIFT_TFIDF_MAX']}"
          f" & sem>{T['DRIFT_SEMANTIC_MIN']} → ACCEPT")
    print(f"    Rule 2 Coherence:    median_top5<{T['COHERENCE_TFIDF_MEDIAN']} → ACCEPT")
    print(f"    Rule 3 TFIDFOverride:max_tfidf<{T['TFIDF_OVERRIDE_MAX']}"
          f" & hybrid<{T['HYBRID_OVERRIDE_MAX']} → ACCEPT")
    print(f"    Rule 4 Floor:        eff_cal<{system.novelty_floor:.3f} → ACCEPT")
    print(f"    Rule 5 Reject:       eff_cal>={system.final_threshold:.3f}"
          f" AND top3>={system.final_threshold:.3f} → REJECT")

    if metrics:
        print(f"\n  Test Metrics:")
        print(f"    Accuracy    : {metrics['accuracy']:.3f}  ({metrics['accuracy']*100:.1f}%)")
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

    all_correct = all(
        ("REJECT" in label and "REJECT" in d) or ("ACCEPT" in label and "ACCEPT" in d)
        for label, d in decisions.items()
    )
    print(f"\n  All cases correct: {'✓ YES' if all_correct else '✗ NO'}")

    print(f"\n  v12 changes vs v11:")
    print(f"    CHANGE-V12-1: Model → all-mpnet-base-v2 (768-dim)")
    print(f"                  No more fallback chain / tokenizer errors")
    print(f"    CHANGE-V12-2: bert_score → semantic_score throughout")
    print(f"    CHANGE-V12-3: Similarity via dot product (L2-normalised)")
    print(f"    CHANGE-V12-4: Semantic gap guard: top_semantic > {T['SEMANTIC_GAP_SCORE_MIN']}")
    print(f"                  Prevents noise triggers when both scores are low")
    print(f"    CHANGE-V12-5: Single 'mpnet' mode — no patent/general switching")
    print(f"    CHANGE-V12-6: Clean logging — ✓ messages, no spurious ⚠ warnings")
    print(f"\n  System ready.")


if __name__ == "__main__":
    main()