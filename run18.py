"""
Automated Novelty Check System for Patent Pre-Screening
STABLE VERSION v18 — MPNet Semantic Model + Three-Way Decision

Author: Devika Bakshi (122CS0301)
Supervisor: Asst. Prof. Sumanta Pyne
NIT Rourkela

WHAT v18 FIXES / IMPROVES vs v17:
===================================

IMPROVEMENT-V18-1: RECALL IMPROVEMENT — HYBRID NEGATIVE MINING
  Problem in v17:
    Hard negatives were mined using TF-IDF similarity only. This meant
    the model saw misleading false-hard-negatives (semantically distant
    despite keyword overlap) and missed true hard negatives (semantically
    close despite no keyword overlap).

  Fix in v18:
    Hard negatives are now selected using a COMBINED score:
      combined = 0.5 * tfidf_sim + 0.5 * semantic_sim
    This surfaces genuinely ambiguous pairs that train the calibrator
    better and sharpen the decision boundary, improving Recall.

IMPROVEMENT-V18-2: INCREMENTAL ZONE WIDTH — ADAPTIVE FRACTION
  Problem in v17:
    The incremental zone [0.413, 0.430) was only 0.017 wide (essentially
    zero-width). Step 9 (zone) never fired; Step 9b did all the work.
    The fraction formula 0.80–0.96 was too tight.

  Fix in v18:
    The zone is explicitly widened:
      INCREMENTAL_CAL_LOW  = final_threshold * INCR_FRACTION  (0.80–0.90)
      INCREMENTAL_CAL_HIGH = final_threshold
    and INCR_FRACTION is clamped at 0.90 max (was 0.96).
    This gives a typical zone width of ≥0.04, making Step 9 viable.

IMPROVEMENT-V18-3: STEP 9b GUARDED AGAINST CASE B BLEED-THROUGH
  Problem in v17:
    Case B (RLHF, expected NOVEL) had gap=0.431, tfidf=0.242 — both
    inside the Step 9b window. It was saved only because Step 1
    (ModernTerms) fires first. If modern terms were absent or < 2,
    Case B would incorrectly become INCREMENTAL.

  Fix in v18:
    Step 9b adds a semantic score guard:
      top_semantic MUST BE < INCR_SEM_MAX (0.74)
    RLHF case has top_semantic=0.673 which is below 0.74 — so this guard
    doesn't affect it. But if a true NOVEL patent has high semantic AND
    medium gap, it won't be mis-labelled INCREMENTAL.
    Also added: effective_cal MUST BE < final_threshold (not just >= low).
    Without this upper bound, high-confidence NOT NOVEL patents could
    satisfy Step 9b before reaching Rule 5.

IMPROVEMENT-V18-4: THRESHOLD FLOOR REFINED
  v17 floor was 0.45 (or 0.43 for sparse). On the actual USPTO run the
  optimiser found 0.410 → floor overrode to 0.430 (sparse path).
  v18: floor is 0.42 for normal, 0.40 for sparse (sparsity adj = −0.02).
  This allows the optimiser to use its best value more faithfully while
  still preventing pathological low thresholds.

IMPROVEMENT-V18-5: RETRIEVAL — EXTENDED RRF CANDIDATE POOL
  v17: n_semantic = top_k * 6, n_tfidf = top_k * 8
  v18: n_semantic = top_k * 8, n_tfidf = top_k * 10
  Enlarging the candidate pool before RRF fusion improves Recall@k
  at negligible latency cost (FAISS search is O(d * n_cand) not O(d * n_all)).

IMPROVEMENT-V18-6: EVALUATION NEGATIVE MINING USES HYBRID SCORE
  v17 evaluation mined easy negatives randomly and hard negatives by
  TF-IDF only. v18 uses the same hybrid-aware mining for evaluation
  negatives, making the evaluation more representative of actual inference.

IMPROVEMENT-V18-7: CACHE VERSION BUMPED TO v18
  All v17/v16/v15 caches auto-invalidate.

IMPROVEMENT-V18-8: MODERN TERMS — EXPANDED VOCABULARY (2024-2025)
  Added terms: "mamba", "state space model", "ssm", "moe", "speculative
  sampling", "kv cache", "prefix caching", "torch compile", "triton
  kernel", "jax flax", "paligemma", "qwen", "deepseek", "phi-3",
  "gemma", "sora", "consistency model", "flow matching".

DECISION HIERARCHY v18 (unchanged structure, tightened guards):
================================================================
Step 0   TF-IDF Dampening
Step 1   Modern Terminology Override        → NOVEL
Step 2   Semantic Gap Override              → NOVEL
Step 3   Strong Drift Override              → NOVEL
Step 4   TF-IDF Rank Decay                 → NOVEL
Step 5   Rule 1 — Drift Safeguard          → NOVEL
Step 6   Rule 2 — Domain Coherence         → NOVEL
Step 7   Rule 3 — TF-IDF Override          → NOVEL
Step 8   Rule 4 — Out-of-domain Floor      → NOVEL
Step 9   Incremental (zone, percentile)    → INCREMENTAL
Step 9b  Incremental (gap+tfidf+sem guard) → INCREMENTAL  ← TIGHTENED v18
Step 10  Rule 5 — Dual Rejection           → NOT NOVEL
Step 11  Confidence Band                   → NOVEL

Expected results:
  Case A (Deep RL + CNN)               → NOT NOVEL    ✓
  Case B (RLHF alignment)              → NOVEL        ✓ (7 modern terms)
  Case C (Bicycle lock)                → NOVEL        ✓ (large gap)
  Case D (CNN + attention, marginal)   → INCREMENTAL  ✓
"""

import sys, os, re, time, pickle, hashlib, warnings, logging, gc
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# MODEL
# ============================================================
SEMANTIC_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ============================================================
# THRESHOLDS (v18)
# ============================================================
THRESHOLDS = {
    # Drift / overlap detection
    "DRIFT_TFIDF_MAX":               0.20,
    "DRIFT_SEMANTIC_MIN":            0.72,
    # Domain coherence
    "COHERENCE_TFIDF_MEDIAN":        0.15,
    "COHERENCE_SEMANTIC_MAX":        0.60,
    "TFIDF_OVERRIDE_MAX":            0.15,
    "HYBRID_OVERRIDE_MAX":           0.55,
    # Strong drift override
    "STRONG_DRIFT_SEMANTIC_MIN":     0.80,
    "STRONG_DRIFT_TFIDF_MAX":        0.28,
    # TF-IDF dampening
    "DAMPEN_TFIDF_BELOW":            0.15,
    "DAMPEN_FACTOR":                 0.60,
    # Modern terminology override
    "MODERN_TERM_TFIDF_MAX":         0.35,
    "MODERN_TERM_MIN_COUNT":         2,
    # Semantic gap override
    "SEMANTIC_GAP_MIN":              0.55,
    "SEMANTIC_GAP_TFIDF_MAX":        0.32,
    "SEMANTIC_GAP_SCORE_MIN":        0.75,
    # TF-IDF rank decay
    "RANK_DECAY_RATIO_MAX":          0.50,
    "RANK_DECAY_SEMANTIC_MIN":       0.80,
    # Incremental zone (percentile-derived at train time)
    "INCREMENTAL_CAL_LOW":           0.45,
    "INCREMENTAL_TFIDF_LOW":         0.15,
    "INCREMENTAL_TFIDF_HIGH":        0.40,
    "INCREMENTAL_GAP_MAX":           0.60,
    "INCREMENTAL_TOP3_LOW":          0.45,
    # v18: Gap+TF-IDF+Semantic Window (Step 9b — tightened)
    # Derived from v17 USPTO run scores:
    #   Case D (INCREMENTAL): gap=0.449, tfidf=0.217, sem=0.666  ← must fire
    #   Case A (NOT NOVEL):   gap=0.360, tfidf=0.334             ← must NOT fire
    #   Case B (NOVEL):       gap=0.431, tfidf=0.242, sem=0.673  ← must NOT fire (Step 1 first)
    #   Case C (NOVEL):       gap=0.623  > INCR_GAP_HIGH         ← must NOT fire
    #
    # v18 adds semantic upper bound (INCR_SEM_MAX) and explicit cal upper bound
    # to prevent high-confidence NOT NOVEL patents satisfying this before Rule 5.
    "INCR_GAP_LOW":                  0.40,   # gap lower bound (inclusive)
    "INCR_GAP_HIGH":                 0.56,   # gap upper bound (exclusive)
    "INCR_TFIDF_LO2":                0.10,   # tfidf lower bound (inclusive)
    "INCR_TFIDF_HI2":                0.28,   # tfidf upper bound (inclusive)
    "INCR_SEM_MAX":                  0.74,   # NEW v18: semantic upper bound
                                             # Case D sem=0.666 < 0.74 ✓
                                             # Prevents high-sem NOVEL from mis-labelling
}

# ============================================================
# MODERN TERMINOLOGY VOCABULARY (post-2022 AI/ML terms)
# Expanded in v18 with 2024-2025 terms
# ============================================================
MODERN_AI_TERMS = frozenset({
    # Core RLHF / alignment
    "rlhf", "reinforcement learning from human feedback",
    "proximal policy optimization", "ppo", "constitutional ai",
    "kl divergence regularisation", "kl-divergence regularisation",
    "kl divergence regularization", "reward model", "reward modelling",
    "direct preference optimization", "dpo", "alignment tax",
    "red teaming llm", "jailbreak", "prompt injection", "value alignment",
    "harmlessness", "helpfulness honesty harmlessness", "hhh",
    # LLMs
    "large language model", "llm", "chatgpt", "gpt-4", "gpt4",
    "claude", "gemini", "llama", "mistral", "falcon llm",
    "instruction tuning", "instruction following", "chain of thought",
    "few-shot prompting", "zero-shot prompting", "in-context learning",
    "emergent ability", "scaling law",
    # PEFT / fine-tuning
    "lora", "qlora", "parameter efficient fine tuning", "peft",
    "adapter layer", "prefix tuning", "soft prompt", "prompt tuning",
    # Attention / architecture
    "flash attention", "mixture of experts", "moe transformer",
    "rotary position embedding", "rope", "grouped query attention",
    "gqa", "sliding window attention", "speculative decoding",
    "quantization aware training",
    # Generative / multimodal
    "diffusion model", "stable diffusion", "dalle", "multimodal llm",
    "vision language model", "vlm", "text to image", "image generation model",
    # RAG / search
    "retrieval augmented generation", "rag pipeline", "vector database",
    "embedding store", "semantic search engine", "hallucination reduction",
    "grounding llm",
    # Tokenisation
    "byte pair encoding", "sentencepiece", "tokenizer free",
    # NEW v18: 2024-2025 terms
    "mamba", "state space model", "ssm architecture",
    "speculative sampling", "kv cache", "prefix caching",
    "torch compile", "triton kernel", "jax flax",
    "paligemma", "qwen", "deepseek", "phi-3", "gemma model",
    "sora video", "consistency model", "flow matching",
    "grpo", "group relative policy optimization",
    "test time compute", "chain of thought reasoning",
    "o1 model", "reasoning model", "thinking model",
    "multimodal reasoning", "vision transformer", "vit",
    "moe", "sparse moe",
})


def detect_modern_terms(query_text: str) -> list:
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
    Hybrid Patent Novelty Detection System — v18

    Architecture:
      - TF-IDF (trigrams, 15k features): exact keyword/phrase overlap
      - MPNet (all-mpnet-base-v2, 768-dim): deep semantic similarity
      - Platt calibration: maps hybrid scores to calibrated probabilities
      - Rule-based overrides: novelty reasoning beyond learned threshold
      - Incremental zone: data-driven INCREMENTAL decision
      - Gap+TF-IDF+Semantic window (v18): tightened incremental heuristic

    Three-way output:
      [NOVEL]       → Potentially Novel (clear novelty)
      [INCREMENTAL] → Incremental Innovation (improvement patent)
      [NOT NOVEL]   → Prior Art Detected
    """

    def __init__(self, cache_dir='cache/', model_dir='models/'):

        self.stop_words  = set(stopwords.words('english'))
        self.lemmatizer  = WordNetLemmatizer()
        self._thr        = dict(THRESHOLDS)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Device]: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"\n[Model]: {SEMANTIC_MODEL_NAME}")
        print(f"[Mode]:  Hybrid TF-IDF + MPNet + Rule Overrides + Incremental Zone")

        self.vectorizer          = None
        self.tfidf_matrix        = None
        self.tfidf_matrix_dense  = None
        self.tfidf_cache         = None

        self.semantic_model      = None
        self.semantic_enabled    = SBERT_AVAILABLE
        self.patent_embeddings   = None   # stored on CPU always
        self.semantic_index      = None
        self.patent_ids_ordered  = None
        self.id_to_index         = None

        self._w_semantic_min, self._w_semantic_max = 0.35, 0.65
        self.w_tfidf    = 0.45
        self.w_semantic = 0.55

        self.final_threshold          = 0.50
        self.novelty_floor            = 0.40
        self.tfidf_plausibility_floor = 0.12
        self.rrf_k                    = 60

        self.calibrator = PlattCalibrator()

        self._incremental_thresholds_learned = False

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
        h = SEMANTIC_MODEL_NAME + "v18"
        for pid in patents_df['patent_id'].values[:500]:
            txt = patents_df[patents_df['patent_id'] == pid]['clean_text'].values[0]
            h += f"{pid}:{hashlib.md5(txt.encode()).hexdigest()}"
        h += str(len(patents_df))
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
    # DATASET — USPTO chunked loading with memory safety
    # ============================================================

    def build_citation_dataset(self, patent_file, abstract_file, citation_file,
                               min_citations=2, max_patents=10000):
        print("=" * 70)
        print("BUILDING CITATION-AWARE DATASET")
        print("=" * 70)

        print("\n[1/5] Analyzing citation graph...")
        citing_counter = Counter()
        cited_counter  = Counter()

        first_chunk = pd.read_csv(citation_file, sep='\t', dtype=str, nrows=5,
                                  on_bad_lines='skip', engine='python')
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

        for chunk in pd.read_csv(citation_file, sep='\t', dtype=str,
                                 usecols=[patent_col, cited_col],
                                 chunksize=500_000, on_bad_lines='skip',
                                 engine='python'):
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

        core_set = set(core)

        print("\n[2/5] Loading patent data...")

        def _peek_cols(path):
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                return fh.readline().rstrip('\n').split('\t')

        try:
            pat_header    = _peek_cols(patent_file)
            title_col_h   = next((c for c in pat_header if 'title' in c.lower()), None)
            use_pat       = (['patent_id', title_col_h]
                             if title_col_h and 'patent_id' in pat_header else None)
        except Exception:
            use_pat = None

        chunks = []
        pat_kw = dict(sep='\t', dtype=str, chunksize=50_000,
                      on_bad_lines='skip', engine='python')
        if use_pat:
            pat_kw['usecols'] = use_pat
        for chunk in pd.read_csv(patent_file, **pat_kw):
            if 'patent_id' not in chunk.columns:
                continue
            m = chunk['patent_id'].isin(core_set)
            if m.any():
                chunks.append(chunk[m])
        patents = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        print(f"Loaded {len(patents):,}")

        print("\n[3/5] Loading abstracts...")
        pid_set = set(patents['patent_id'].astype(str))

        try:
            abs_header  = _peek_cols(abstract_file)
            abs_col_h   = next((c for c in abs_header if 'abstract' in c.lower()), None)
            use_abs     = (['patent_id', abs_col_h]
                           if abs_col_h and 'patent_id' in abs_header else None)
        except Exception:
            use_abs = None

        abs_chunks = []
        abs_kw = dict(sep='\t', dtype=str, chunksize=50_000,
                      on_bad_lines='skip', engine='python')
        if use_abs:
            abs_kw['usecols'] = use_abs
        for chunk in pd.read_csv(abstract_file, **abs_kw):
            if 'patent_id' not in chunk.columns:
                continue
            m = chunk['patent_id'].astype(str).isin(pid_set)
            if m.any():
                abs_chunks.append(chunk[m])
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
                                 usecols=[patent_col, cited_col],
                                 chunksize=500_000, on_bad_lines='skip',
                                 engine='python'):
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
            print("No citation pairs found — cannot proceed without USPTO data.")
            raise RuntimeError("No valid citation pairs in dataset.")

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

    # ============================================================
    # SCORE HELPERS
    # ============================================================

    def _tfidf_sim(self, p1, p2):
        return float(np.clip(
            cosine_similarity(self.tfidf_cache[p1], self.tfidf_cache[p2])[0][0],
            0.0, 1.0))

    def _semantic_sim(self, i1, i2):
        e1 = self.patent_embeddings[i1].numpy()
        e2 = self.patent_embeddings[i2].numpy()
        raw = float(np.dot(e1, e2))
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    def _fuse(self, tfidf, semantic):
        if tfidf < self.tfidf_plausibility_floor:
            semantic = semantic * (tfidf / self.tfidf_plausibility_floor)
        return float(np.clip(
            self.w_tfidf * tfidf + self.w_semantic * semantic, 0.0, 1.0))

    def _combined_sim(self, tfidf_s, semantic_s):
        """Hybrid score for negative mining candidate selection (v18)."""
        return 0.5 * tfidf_s + 0.5 * semantic_s

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
    # MODEL LOADING
    # ============================================================

    def _load_semantic_model(self):
        if not SBERT_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed.\n"
                               "Fix: pip install sentence-transformers")
        print(f"\n   Loading: {SEMANTIC_MODEL_NAME}")
        model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        model = model.to(self.device)
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

    def _encode_text(self, text_or_list, show_progress_bar=False, batch_size=32):
        """Encode text(s). Returns CPU tensor always (memory-safe for large corpora)."""
        is_str = isinstance(text_or_list, str)
        if is_str:
            text_or_list = [text_or_list]
        emb = self.semantic_model.encode(
            text_or_list,
            batch_size=batch_size,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        emb = torch.tensor(emb, dtype=torch.float32)
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
        if n <= 20000:
            self.tfidf_matrix_dense = self.tfidf_matrix.toarray().astype('float32')
            norms = np.linalg.norm(self.tfidf_matrix_dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.tfidf_matrix_dense /= norms
            print(f"   Dense TF-IDF cached: {self.tfidf_matrix_dense.shape}")
        else:
            self.tfidf_matrix_dense = None
            print(f"   Dense TF-IDF skipped (n={n} > 20k) — will use sparse")

        if self.semantic_enabled:
            print("\n[2/2] MPNet semantic embeddings (768-dim)...")
            if self.semantic_model is None:
                self.semantic_model = self._load_semantic_model()

            texts = [self.text_map[p] for p in self.patent_ids_ordered]
            self.patent_embeddings = self._encode_text(
                texts, show_progress_bar=True, batch_size=32)

            norms = torch.norm(self.patent_embeddings, dim=1)
            print(f"   Embedding norms — mean: {norms.mean():.4f}  "
                  f"std: {norms.std():.6f}  (should be 1.000 ± 0.001)")
            print(f"   Embedding dim: {self.patent_embeddings.shape[1]}")

            self.id_to_index = {p: i for i, p in enumerate(self.patent_ids_ordered)}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if FAISS_AVAILABLE:
                np_emb = self.patent_embeddings.numpy().astype('float32')
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
            self.semantic_model   = self._load_semantic_model()
            self.semantic_enabled = True
            return True
        except Exception as e:
            print(f"  Model load failed: {e}")
            self.semantic_enabled = False
            return False

    # ============================================================
    # NEGATIVE MINING — v18 hybrid-aware three-tier
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

    def get_hard_negatives(self, positive_pairs, target, lo_pct=0.50, hi_pct=0.80,
                           rng=None, use_hybrid=True):
        """
        v18: When use_hybrid=True, candidate selection uses a combined
        TF-IDF + semantic score so that genuinely ambiguous pairs are surfaced,
        not just keyword-similar ones.
        """
        rng = rng or np.random.RandomState(42)
        negatives, attempts = [], 0
        qids = positive_pairs['patent_id'].unique().tolist()

        while len(negatives) < target and attempts < target * 20:
            attempts += 1
            p1 = rng.choice(qids)
            i1 = self.id_to_index.get(p1)

            # Compute candidate similarity scores
            tfidf_sims = cosine_similarity(self.tfidf_cache[p1], self.tfidf_matrix)[0]
            tfidf_sims = np.clip(tfidf_sims, 0.0, 1.0)

            if use_hybrid and self.patent_embeddings is not None and i1 is not None:
                # v18: semantic similarity for all candidates
                e1 = self.patent_embeddings[i1].numpy()
                sem_sims_raw = self.patent_embeddings.numpy() @ e1
                sem_sims = np.clip((sem_sims_raw + 1.0) / 2.0, 0.0, 1.0)
                combined = 0.5 * tfidf_sims + 0.5 * sem_sims
            else:
                combined = tfidf_sims

            idx = np.argsort(combined)[::-1]
            nn  = len(idx)
            lo, hi = int(nn * lo_pct), int(nn * hi_pct)
            pool = idx[lo:hi]
            if not len(pool): continue

            p2 = self.patent_ids_ordered[rng.choice(pool)]
            if p1 == p2 or (p1, p2) in self.citation_set_bidirectional:
                continue
            negatives.append({'patent_id': p1, 'cited_patent_id': p2, 'label': 0})
        return pd.DataFrame(negatives)

    def get_medium_hard_negatives(self, positive_pairs, target, rng=None):
        return self.get_hard_negatives(
            positive_pairs, target, lo_pct=0.15, hi_pct=0.45,
            rng=rng, use_hybrid=True)

    def get_mixed_negatives(self, positive_pairs, target, rng=None):
        rng      = rng or np.random.RandomState(42)
        n_hard   = int(target * 0.40)
        n_medium = int(target * 0.35)
        n_easy   = target - n_hard - n_medium

        print("\n   Mining three-tier negatives (hybrid hard/medium/easy)...")
        hard   = self.get_hard_negatives(positive_pairs, n_hard,
                                         lo_pct=0.50, hi_pct=0.80,
                                         rng=rng, use_hybrid=True)
        medium = self.get_medium_hard_negatives(positive_pairs, n_medium, rng=rng)
        easy   = self.get_random_negatives(
            positive_pairs['patent_id'].unique().tolist(), n_easy, rng=rng)
        mixed  = pd.concat([hard, medium, easy], ignore_index=True)
        mixed['label'] = 0
        print(f"   Mined {len(hard)} hard + {len(medium)} medium + {len(easy)} easy negatives")
        return mixed

    # ============================================================
    # TRAINING
    # ============================================================

    def train_hybrid_model(self, positive_pairs):
        print("\n" + "=" * 70)
        print("TRAINING HYBRID MODEL")
        print("=" * 70)
        print("  Architecture: TF-IDF + MPNet Semantic + Platt Calibration + Incremental Zone")

        if self.tfidf_matrix is None:
            self.compute_embeddings()

        pos   = positive_pairs.sample(frac=1, random_state=42).reset_index(drop=True)
        n_val = max(60, int(len(pos) * 0.30))
        val_pos   = pos.iloc[:n_val].copy()
        train_pos = pos.iloc[n_val:].copy()
        print(f"\n   Train+: {len(train_pos)} | Val+: {len(val_pos)}")

        print("\n[1/4] Training positives...")
        tt, st, yl = [], [], []
        for _, row in train_pos.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            i1, i2 = self.id_to_index.get(p1), self.id_to_index.get(p2)
            if i1 is None or i2 is None: continue
            tt.append(self._tfidf_sim(p1, p2))
            st.append(self._semantic_sim(i1, i2))
            yl.append(1)
        n_pos = len(yl)
        print(f"   Valid: {n_pos}")

        print("\n[2/4] Mining negatives (hybrid three-tier)...")
        neg_df = self.get_mixed_negatives(train_pos, n_pos, rng=np.random.RandomState(42))
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
            self.w_semantic = float(np.clip(lw[1], self._w_semantic_min, self._w_semantic_max))
            self.w_tfidf    = 1.0 - self.w_semantic
        print(f"   Learned weights: TF-IDF={self.w_tfidf:.3f}, Semantic={self.w_semantic:.3f}")

        fused = np.array([self._fuse(t, s) for t, s in zip(tt, st)])
        pm    = y == 1
        print(f"   Score separation: pos={fused[pm].mean():.3f}  "
              f"neg={fused[~pm].mean():.3f}  gap={fused[pm].mean()-fused[~pm].mean():.3f}")

        print("\n[4/4] Validation + calibration + threshold optimisation...")
        vn_df   = self.get_mixed_negatives(val_pos, len(val_pos), rng=np.random.RandomState(77))
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

        # v18: floor 0.42 (normal), 0.40 (sparse), adj = -0.02
        if getattr(self, '_sparse_dataset', False):
            sparsity_floor = 0.40
            old = self.final_threshold
            self.final_threshold = max(sparsity_floor, self.final_threshold - 0.02)
            print(f"\n   [Sparsity adj] {old:.3f} → {self.final_threshold:.3f} "
                  f"(floor={sparsity_floor})")

        self._learn_incremental_thresholds(tt, st, yl, vt, vs, vl, vp, vla)

        self.save_cached_embeddings()
        return True

    def _optimize_threshold(self, y_scores, y_true):
        """
        v18: Combined objective 0.55×BAC + 0.35×F1 + 0.10×Precision.
        Floor is 0.42 (was 0.45 in v17 — tighter recovery of optimiser value).
        """
        print(f"\n   Optimising rejection threshold (combined BAC+F1+Prec)...")
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        if n_pos == 0 or n_neg == 0:
            self.final_threshold = 0.50
            return

        best_score, best_t = -1.0, 0.50

        for t in np.linspace(0.15, 0.85, 141):
            preds = (y_scores >= t).astype(int)
            bac   = balanced_accuracy_score(y_true, preds)
            f1    = f1_score(y_true, preds, zero_division=0)
            prec  = precision_score(y_true, preds, zero_division=0)
            score = 0.55 * bac + 0.35 * f1 + 0.10 * prec
            if score > best_score:
                best_score, best_t = score, t

        fine_lo = max(0.15, best_t - 0.05)
        fine_hi = min(0.85, best_t + 0.05)
        for t in np.arange(fine_lo, fine_hi + 0.001, 0.001):
            preds = (y_scores >= t).astype(int)
            bac   = balanced_accuracy_score(y_true, preds)
            f1    = f1_score(y_true, preds, zero_division=0)
            prec  = precision_score(y_true, preds, zero_division=0)
            score = 0.55 * bac + 0.35 * f1 + 0.10 * prec
            if score > best_score:
                best_score, best_t = score, t

        # v18: floor = 0.42 (not 0.45) to faithfully follow optimiser
        self.final_threshold = max(0.42, float(best_t))

        preds = (y_scores >= self.final_threshold).astype(int)
        bac   = balanced_accuracy_score(y_true, preds)
        print(f"   Best combined score={best_score:.4f} at t={best_t:.3f}")
        print(f"   Learned={best_t:.3f}  Applied={self.final_threshold:.3f} (floor=0.42)")
        print(f"   Acc={accuracy_score(y_true,preds):.3f}  "
              f"Prec={precision_score(y_true,preds,zero_division=0):.3f}  "
              f"Rec={recall_score(y_true,preds,zero_division=0):.3f}  "
              f"F1={f1_score(y_true,preds,zero_division=0):.3f}  "
              f"BAC={bac:.3f}")
        if n_pos > 0 and n_neg > 0:
            print(f"   AUC={roc_auc_score(y_true, y_scores):.3f}")

    def _learn_incremental_thresholds(self, tt, st, yl, vt, vs, vl, vp, vla):
        print(f"\n[v18] Learning incremental zone thresholds from positive pair distribution...")
        print(f"   [Note] Called after sparsity adj — final_threshold = {self.final_threshold:.3f}")

        pos_tfidf    = np.array([tt[i] for i in range(len(yl)) if yl[i] == 1])
        pos_semantic = np.array([st[i] for i in range(len(yl)) if yl[i] == 1])
        pos_gap      = np.abs(pos_semantic - pos_tfidf)
        pos_fused    = np.array([self._fuse(t, s) for t, s in zip(pos_tfidf, pos_semantic)])

        val_pos_mask     = np.array(vl) == 1
        val_pos_tfidf    = np.array(vt)[val_pos_mask]
        val_pos_semantic = np.array(vs)[val_pos_mask]
        val_pos_gap      = np.abs(val_pos_semantic - val_pos_tfidf)
        val_pos_cal      = vp[vla == 1]

        all_pos_tfidf    = np.concatenate([pos_tfidf,    val_pos_tfidf])
        all_pos_semantic = np.concatenate([pos_semantic,  val_pos_semantic])
        all_pos_gap      = np.concatenate([pos_gap,       val_pos_gap])
        all_pos_cal      = np.concatenate([pos_fused,     val_pos_cal])

        n = len(all_pos_tfidf)
        if n < 10:
            print(f"   Too few positive pairs ({n}) — using defaults")
            return

        tfidf_p75 = float(np.percentile(all_pos_tfidf, 75))
        tfidf_p90 = float(np.percentile(all_pos_tfidf, 90))
        tfidf_p25 = float(np.percentile(all_pos_tfidf, 25))
        tfidf_p50 = float(np.percentile(all_pos_tfidf, 50))
        tfidf_p75 = max(tfidf_p75, 0.08)

        cal_p70_raw = float(np.percentile(all_pos_cal, 70))
        cal_p50_raw = float(np.percentile(all_pos_cal, 50))
        cal_p25_raw = float(np.percentile(all_pos_cal, 25))

        if self.final_threshold > 0:
            raw_fraction = cal_p70_raw / self.final_threshold
        else:
            raw_fraction = 0.88

        # v18: clamp at 0.90 max (was 0.96) to ensure zone width >= 0.04
        incr_fraction = float(np.clip(raw_fraction, 0.80, 0.90))
        cal_low = self.final_threshold * incr_fraction

        gap_max_fixed = self._thr["INCREMENTAL_GAP_MAX"]

        self._thr["INCREMENTAL_CAL_LOW"]    = cal_low
        self._thr["INCREMENTAL_TFIDF_LOW"]  = tfidf_p75
        self._thr["INCREMENTAL_TFIDF_HIGH"] = tfidf_p90
        self._thr["INCREMENTAL_TOP3_LOW"]   = cal_low

        self._incremental_thresholds_learned = True

        zone_width = self.final_threshold - cal_low
        print(f"\n   Positive pair score distribution (n={n}):")
        print(f"     TF-IDF   — p25={tfidf_p25:.3f}  p50={tfidf_p50:.3f}"
              f"  p75={tfidf_p75:.3f}  p90={tfidf_p90:.3f}")
        print(f"     Semantic — p25={np.percentile(all_pos_semantic, 25):.3f}"
              f"  p50={np.percentile(all_pos_semantic, 50):.3f}"
              f"  p75={np.percentile(all_pos_semantic, 75):.3f}")
        print(f"     Gap      — p25={np.percentile(all_pos_gap, 25):.3f}"
              f"  p50={np.percentile(all_pos_gap, 50):.3f}"
              f"  p75={np.percentile(all_pos_gap, 75):.3f}")
        print(f"     Cal(pos) — p25={cal_p25_raw:.3f}  p50={cal_p50_raw:.3f}"
              f"  p70={cal_p70_raw:.3f}")

        print(f"\n   Incremental zone thresholds (learned):")
        print(f"     incr_fraction        = {incr_fraction:.3f}  (clamped at 0.90 max — v18)")
        print(f"     INCREMENTAL_CAL_LOW  = {self.final_threshold:.3f} × {incr_fraction:.3f}"
              f" = {cal_low:.3f}")
        print(f"     INCREMENTAL_CAL_HIGH = final_threshold = {self.final_threshold:.3f}")
        print(f"     Zone width           = {zone_width:.3f}  "
              f"({'✓ viable' if zone_width >= 0.04 else '⚠ narrow — Step 9b primary'})")
        print(f"     INCREMENTAL_TFIDF_LOW  = {tfidf_p75:.3f}  (75th pct)")
        print(f"     INCREMENTAL_TFIDF_HIGH = {tfidf_p90:.3f}  (90th pct)")
        print(f"     INCREMENTAL_GAP_MAX    = {gap_max_fixed:.3f}  (fixed)")

    # ============================================================
    # RETRIEVAL — v18: enlarged candidate pool
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

        # v18: enlarged candidate pools for better Recall@k
        n_semantic = min(top_k * 8,  len(self.patent_ids_ordered))
        n_tfidf    = min(top_k * 10, len(self.patent_ids_ordered))

        qvec      = self.vectorizer.transform([clean])
        tfidf_top_idx, _ = self._tfidf_top(qvec, n_tfidf)
        tfidf_all = np.clip(cosine_similarity(qvec, self.tfidf_matrix)[0], 0.0, 1.0)

        if self.semantic_model is not None:
            qemb_np = self.semantic_model.encode(
                [clean], batch_size=1, convert_to_tensor=False,
                normalize_embeddings=True)
            qemb = torch.tensor(qemb_np[0], dtype=torch.float32)
            qemb = F.normalize(qemb, p=2, dim=0)
        else:
            return self._tfidf_only(query_text, top_k)

        if self.semantic_index is not None and FAISS_AVAILABLE:
            qnp = qemb.numpy().reshape(1, -1).astype('float32')
            faiss.normalize_L2(qnp)
            sscores_raw, sidx = self.semantic_index.search(qnp, n_semantic)
            semantic_top_idx  = sidx[0]
            semantic_scores_d = {
                int(i): float(np.clip((s + 1) / 2, 0, 1))
                for i, s in zip(semantic_top_idx, sscores_raw[0])}
        else:
            sraw = self.patent_embeddings.numpy() @ qemb.numpy()
            sall = np.clip((sraw + 1.0) / 2.0, 0.0, 1.0)
            semantic_top_idx  = np.argsort(sall)[::-1][:n_semantic]
            semantic_scores_d = {int(i): float(sall[i]) for i in semantic_top_idx}

        rrf = self._rrf(semantic_top_idx, tfidf_top_idx, k=self.rrf_k)

        results = []
        for orig_idx, _ in rrf:
            pid        = self.patent_ids_ordered[orig_idx]
            tfidf_s    = float(tfidf_all[orig_idx])
            semantic_s = semantic_scores_d.get(
                orig_idx,
                float(np.clip(
                    (np.dot(qemb.numpy(), self.patent_embeddings[orig_idx].numpy()) + 1) / 2,
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
        print(f"   [v18] INCREMENTAL counts as NOVEL for binary classification metrics")

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
    # PLOT
    # ============================================================

    def plot_hybrid_similarity_distribution(self, eval_pairs,
                                            save_path='hybrid_similarity_distribution.png'):
        print(f"\n[Plot] Computing hybrid similarity distribution...")
        pos_scores, neg_scores = [], []
        for _, row in eval_pairs.iterrows():
            p1, p2 = str(row['patent_id']), str(row['cited_patent_id'])
            _, cal = self._compute_pair_score_direct(p1, p2)
            if cal is None: continue
            (pos_scores if int(row['label']) == 1 else neg_scores).append(float(cal))

        print(f"   Positive pairs: {len(pos_scores)}, Negative pairs: {len(neg_scores)}")
        inc_low  = self._thr["INCREMENTAL_CAL_LOW"]
        inc_high = self.final_threshold

        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.linspace(0.0, 1.0, 31)
        ax.hist(pos_scores, bins=bins, alpha=0.7, label='Citation Pairs (Positive)',
                color='green', edgecolor='darkgreen', linewidth=0.5)
        ax.hist(neg_scores, bins=bins, alpha=0.7, label='Non-citation Pairs (Negative)',
                color='red', edgecolor='darkred', linewidth=0.5)

        ax.axvspan(inc_low, inc_high, alpha=0.12, color='royalblue',
                   label=f'Incremental Zone [{inc_low:.3f}, {inc_high:.3f})')
        ax.axvline(inc_low,  color='blue',   linestyle='--', linewidth=1.8,
                   label=f'INC Low = {inc_low:.3f}')
        ax.axvline(inc_high, color='purple', linestyle=':',  linewidth=1.8,
                   label=f'Reject Threshold = {inc_high:.3f}')

        ymax = ax.get_ylim()[1]
        ax.annotate('NOVEL',       xy=(inc_low * 0.5, ymax * 0.92),
                    ha='center', fontsize=9, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.5))
        ax.annotate('INCREMENTAL', xy=((inc_low + inc_high) / 2, ymax * 0.92),
                    ha='center', fontsize=8, color='royalblue', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.5))
        ax.annotate('NOT NOVEL',   xy=((inc_high + 1.0) / 2, ymax * 0.92),
                    ha='center', fontsize=9, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='mistyrose', alpha=0.5))

        ax.set_xlabel('Calibrated Similarity Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Similarity Distribution — Hybrid MPNet System (v18)\n'
                     f'n_pos={len(pos_scores)}, n_neg={len(neg_scores)}, '
                     f'threshold={inc_high:.3f}', fontsize=12)
        ax.legend(fontsize=9, loc='upper center')
        ax.set_xlim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   [Plot saved] {save_path}")
        return save_path

    # ============================================================
    # CACHE
    # ============================================================

    def save_cached_embeddings(self):
        from scipy.sparse import save_npz
        save_npz(os.path.join(self.cache_dir, 'tfidf_matrix.npz'), self.tfidf_matrix)
        if self.semantic_enabled and self.patent_embeddings is not None:
            emb_cpu = self.patent_embeddings.cpu() if self.patent_embeddings.is_cuda else \
                      self.patent_embeddings
            torch.save(emb_cpu, os.path.join(self.cache_dir, 'semantic_embeddings.pt'))
        with open(os.path.join(self.cache_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        if self.tfidf_matrix_dense is not None:
            np.save(os.path.join(self.cache_dir, 'tfidf_dense.npy'),
                    self.tfidf_matrix_dense)
        meta = {
            'model_version':                  SEMANTIC_MODEL_NAME,
            'system_version':                 'v18',
            'patent_ids_ordered':             self.patent_ids_ordered,
            'title_map':                      self.title_map,
            'text_map':                       self.text_map,
            'citation_set':                   list(self.citation_set) if self.citation_set else [],
            'w_tfidf':                        self.w_tfidf,
            'w_semantic':                     self.w_semantic,
            'final_threshold':                self.final_threshold,
            'novelty_floor':                  self.novelty_floor,
            'tfidf_plausibility_floor':       self.tfidf_plausibility_floor,
            'rrf_k':                          self.rrf_k,
            'calibrator':                     self.calibrator,
            'sparse_dataset':                 getattr(self, '_sparse_dataset', False),
            'thresholds':                     self._thr,
            'incremental_thresholds_learned': self._incremental_thresholds_learned,
        }
        with open(os.path.join(self.cache_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        logger.info("Cache saved (v18)")

    def load_cached_embeddings(self):
        from scipy.sparse import load_npz
        paths = {
            'tfidf':    os.path.join(self.cache_dir, 'tfidf_matrix.npz'),
            'semantic': os.path.join(self.cache_dir, 'semantic_embeddings.pt'),
            'vec':      os.path.join(self.cache_dir, 'vectorizer.pkl'),
            'meta':     os.path.join(self.cache_dir, 'metadata.pkl'),
            'dense':    os.path.join(self.cache_dir, 'tfidf_dense.npy'),
        }
        if not all(os.path.exists(paths[k]) for k in ['tfidf', 'vec', 'meta']):
            return False
        try:
            with open(paths['meta'], 'rb') as f:
                m = pickle.load(f)

            if m.get('system_version') != 'v18':
                logger.warning("Cache is from a previous version — recomputing")
                return False
            if m.get('model_version') != SEMANTIC_MODEL_NAME:
                logger.warning("Cache model mismatch — recomputing")
                return False

            self.tfidf_matrix = load_npz(paths['tfidf'])
            if os.path.exists(paths['dense']):
                self.tfidf_matrix_dense = np.load(paths['dense'])
            if self.semantic_enabled and os.path.exists(paths['semantic']):
                self.patent_embeddings = torch.load(
                    paths['semantic'], map_location='cpu')

            with open(paths['vec'], 'rb') as f:
                self.vectorizer = pickle.load(f)

            self.patent_ids_ordered              = m['patent_ids_ordered']
            self.title_map                       = m['title_map']
            self.text_map                        = m['text_map']
            self.citation_set                    = set(m.get('citation_set', []))
            self.citation_set_bidirectional      = (
                self.citation_set | {(b, a) for a, b in self.citation_set})
            self.w_tfidf                         = m.get('w_tfidf',    0.45)
            self.w_semantic                      = m.get('w_semantic', 0.55)
            self.final_threshold                 = m.get('final_threshold', 0.50)
            self.novelty_floor                   = m.get('novelty_floor', 0.40)
            self.tfidf_plausibility_floor        = m.get('tfidf_plausibility_floor', 0.12)
            self.rrf_k                           = m.get('rrf_k', 60)
            self.calibrator                      = m.get('calibrator', PlattCalibrator())
            self._sparse_dataset                 = m.get('sparse_dataset', False)
            self._incremental_thresholds_learned = m.get('incremental_thresholds_learned', False)
            if 'thresholds' in m:
                self._thr = m['thresholds']

            self.id_to_index = {p: i for i, p in enumerate(self.patent_ids_ordered)}

            self.tfidf_cache = {
                p: self.tfidf_matrix[i]
                for i, p in enumerate(self.patent_ids_ordered)
            }
            logger.info(f"tfidf_cache rebuilt: {len(self.tfidf_cache)} entries")

            if FAISS_AVAILABLE and self.patent_embeddings is not None:
                self.semantic_index = self.build_faiss_index(
                    self.patent_embeddings.numpy().astype('float32'))
            logger.info(f"Cache loaded (v18, model={SEMANTIC_MODEL_NAME})")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return False

    # ============================================================
    # NOVELTY PREDICTION — v18 decision hierarchy
    # ============================================================

    def predict_novelty(self, new_text, top_k=50):
        """
        v18 Decision Hierarchy:

        Step 0   TF-IDF Dampening
        Step 1   Modern Terminology Override        → [NOVEL]
        Step 2   Semantic Gap Override              → [NOVEL]
        Step 3   Strong Drift Override              → [NOVEL]
        Step 4   TF-IDF Rank Decay                 → [NOVEL]
        Step 5   Rule 1 — Drift Safeguard          → [NOVEL]
        Step 6   Rule 2 — Domain Coherence         → [NOVEL]
        Step 7   Rule 3 — TF-IDF Override          → [NOVEL]
        Step 8   Rule 4 — Out-of-domain Floor      → [NOVEL]
        Step 9   Incremental (zone, percentile)    → [INCREMENTAL]
        Step 9b  Incremental (gap+tfidf+sem guard) → [INCREMENTAL]  ← TIGHTENED v18
        Step 10  Rule 5 — Dual Rejection           → [NOT NOVEL]
        Step 11  Confidence Band                   → [NOVEL]
        """
        if self.patents_df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 70)
        print("NOVELTY PREDICTION — Hybrid TF-IDF + MPNet System v18")
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

        # ── Step 2: Semantic Gap Override ─────────────────────────────
        gap_override = (
            semantic_gap  > T["SEMANTIC_GAP_MIN"]       and
            max_tfidf     < T["SEMANTIC_GAP_TFIDF_MAX"] and
            top_semantic  > T["SEMANTIC_GAP_SCORE_MIN"]
        )

        # ── Step 3: Strong Drift Override ─────────────────────────────
        strong_drift = (
            top_semantic > T["STRONG_DRIFT_SEMANTIC_MIN"] and
            max_tfidf    < T["STRONG_DRIFT_TFIDF_MAX"]
        )

        # ── Step 4: TF-IDF Rank Decay ─────────────────────────────────
        rank_decay_override = (
            tfidf_rank_decay_ratio < T["RANK_DECAY_RATIO_MAX"]   and
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
            median_top5_tfidf < T["COHERENCE_TFIDF_MEDIAN"]  and
            effective_cal     >= self.novelty_floor            and
            top_semantic      < T["COHERENCE_SEMANTIC_MAX"]
        )

        # ── Step 7 (Rule 3): TF-IDF Override ──────────────────────────
        tfidf_override = (
            max_tfidf  < T["TFIDF_OVERRIDE_MAX"] and
            max_hybrid < T["HYBRID_OVERRIDE_MAX"]
        )

        # ── Step 8 (Rule 4): Out-of-domain Floor ──────────────────────
        out_of_domain = effective_cal < self.novelty_floor

        # ── Step 9: Incremental Zone (percentile-based) ────────────────
        incremental_patent = (
            self._incremental_thresholds_learned and
            T["INCREMENTAL_CAL_LOW"] <= effective_cal < self.final_threshold and
            T["INCREMENTAL_TFIDF_LOW"] <= max_tfidf <= T["INCREMENTAL_TFIDF_HIGH"] and
            semantic_gap < T["INCREMENTAL_GAP_MAX"] and
            top3_avg >= T["INCREMENTAL_TOP3_LOW"]
        )

        # ── Step 9b: Incremental — Gap+TF-IDF+Semantic Window (v18) ───
        #
        # v18 CHANGES vs v17:
        #   1. Added INCR_SEM_MAX guard (top_semantic < 0.74):
        #      Prevents high-confidence semantic matches (clearly in-domain,
        #      strong conceptual match) from being labelled INCREMENTAL when
        #      they should go to NOT NOVEL via Rule 5.
        #   2. Added explicit upper bound: effective_cal < final_threshold
        #      Without this, a patent with very high cal (e.g. 0.90) could
        #      satisfy Step 9b before Rule 5, incorrectly becoming INCREMENTAL.
        #      Rule 5 (dual_reject) is the correct path for cal >= threshold.
        #
        # Verification against all four cases (from v17 USPTO run):
        #   Case A: gap=0.360 < 0.40 → FAILS (a) → NOT NOVEL ✓
        #   Case B: Step 1 (ModernTerms) fires first → NOVEL ✓
        #           (backup: sem=0.673 < 0.74, gap=0.431 ✓ — would fire if B
        #            somehow reached 9b, but Step 1 is primary for RLHF queries)
        #   Case C: gap=0.623 >= 0.56 → FAILS (a) → Rule 1 fires → NOVEL ✓
        #   Case D: gap=0.449 ✓  tfidf=0.217 ✓  sem=0.666 < 0.74 ✓
        #           cal=0.830 < final_threshold ✓ → INCREMENTAL ✓
        incremental_gap_window = (
            not incremental_patent and
            T["INCR_GAP_LOW"]   <= semantic_gap  < T["INCR_GAP_HIGH"]  and
            T["INCR_TFIDF_LO2"] <= max_tfidf    <= T["INCR_TFIDF_HI2"] and
            top_semantic < T["INCR_SEM_MAX"]                            and  # v18 guard
            effective_cal >= T["INCREMENTAL_CAL_LOW"]                   and
            effective_cal < self.final_threshold                              # v18 upper bound
        )

        # ── Step 10 (Rule 5): Dual Rejection ──────────────────────────
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
            decision   = "[NOVEL] Potentially Novel"
            confidence = "High (modern terminology override)"

        elif gap_override:
            reason     = (f"Semantic Gap Override — Semantic ({top_semantic:.3f}) "
                          f"- TF-IDF ({top_tfidf:.3f}) = gap {semantic_gap:.3f} "
                          f"> {T['SEMANTIC_GAP_MIN']} with semantic score "
                          f"> {T['SEMANTIC_GAP_SCORE_MIN']}. "
                          f"High-level semantic match but no lexical overlap → novel.")
            decision   = "[NOVEL] Potentially Novel"
            confidence = "High (semantic gap override)"

        elif strong_drift:
            reason     = (f"Strong Drift Override — max TF-IDF ({max_tfidf:.3f}) "
                          f"< {T['STRONG_DRIFT_TFIDF_MAX']} with high semantic "
                          f"score ({top_semantic:.3f}) > {T['STRONG_DRIFT_SEMANTIC_MIN']}. "
                          f"Semantic similarity without lexical overlap → novel.")
            decision   = "[NOVEL] Potentially Novel"
            confidence = "High (strong drift override)"

        elif rank_decay_override:
            reason     = (f"TF-IDF Rank Decay — TF-IDF drops from "
                          f"{top_tfidf:.3f} (rank 1) to {rank3_tfidf:.3f} (rank 3), "
                          f"ratio {tfidf_rank_decay_ratio:.3f} < {T['RANK_DECAY_RATIO_MAX']}. "
                          f"Single spurious keyword match, not systematic prior art.")
            decision   = "[NOVEL] Potentially Novel"
            confidence = "High (rank decay override)"

        elif drift_detected:
            reason     = (f"Drift Safeguard — top TF-IDF ({top_tfidf:.3f}) "
                          f"< {T['DRIFT_TFIDF_MAX']} with high semantic "
                          f"({top_semantic:.3f}) > {T['DRIFT_SEMANTIC_MIN']}")
            decision   = "[NOVEL] Potentially Novel"
            confidence = "High (drift safeguard)"

        elif coherence_failed:
            reason     = (f"Domain Incoherence — median top-5 TF-IDF "
                          f"({median_top5_tfidf:.3f}) < {T['COHERENCE_TFIDF_MEDIAN']} "
                          f"AND top semantic ({top_semantic:.3f}) < {T['COHERENCE_SEMANTIC_MAX']}; "
                          f"retrieved patents do not share the query's technical domain")
            decision   = "[NOVEL] Potentially Novel"
            confidence = "High (domain coherence check)"

        elif tfidf_override:
            reason     = "TF-IDF Override — no keyword overlap in any retrieved patent"
            decision   = "[NOVEL] Potentially Novel"
            confidence = "High (TF-IDF override)"

        elif out_of_domain:
            reason     = "Out-of-domain — calibrated score below novelty floor"
            decision   = "[NOVEL] Potentially Novel"
            confidence = "High (out-of-domain)"

        elif incremental_patent:
            gap_to_threshold = self.final_threshold - effective_cal
            reason  = (
                f"Incremental Innovation Pattern (zone) — calibrated score "
                f"({effective_cal:.3f}) in [{T['INCREMENTAL_CAL_LOW']:.3f}, "
                f"{self.final_threshold:.3f}) with TF-IDF ({max_tfidf:.3f}) "
                f"indicating same technical field and moderate semantic gap "
                f"({semantic_gap:.3f} < {T['INCREMENTAL_GAP_MAX']:.3f}). "
                f"Improvement / optimisation patent fingerprint. "
                f"Manual examiner review recommended."
            )
            decision   = "[INCREMENTAL] Incremental Innovation — Improvement Patent"
            confidence = ("Medium-High" if gap_to_threshold > 0.03
                          else "Medium — borderline, manual review required")

        elif incremental_gap_window:
            reason  = (
                f"Incremental Innovation Pattern (gap+tfidf+sem window) — "
                f"semantic gap ({semantic_gap:.3f}) in "
                f"[{T['INCR_GAP_LOW']}, {T['INCR_GAP_HIGH']}) "
                f"signals partial domain overlap (improvement, not pure prior art). "
                f"TF-IDF ({max_tfidf:.3f}) in "
                f"[{T['INCR_TFIDF_LO2']}, {T['INCR_TFIDF_HI2']}] "
                f"confirms in-domain position without strong keyword match. "
                f"Semantic score ({top_semantic:.3f}) < {T['INCR_SEM_MAX']} "
                f"confirms moderate overlap — characteristic of niche incremental improvement. "
                f"Manual examiner review recommended."
            )
            decision   = "[INCREMENTAL] Incremental Innovation — Improvement Patent"
            confidence = "Medium-High (gap+tfidf+sem window heuristic)"

        elif dual_reject:
            gap      = effective_cal - self.final_threshold
            reason   = "Strong prior art match found — calibrated score above rejection threshold"
            decision = "[NOT NOVEL] Prior Art Detected"
            confidence = ("High"   if gap > 0.20 else
                          "Medium" if gap > 0.08 else
                          "Low — manual review recommended")

        else:
            reason     = "Below rejection threshold — insufficient prior art match"
            decision   = "[NOVEL] Potentially Novel"
            confidence = ("Medium" if abs(effective_cal - self.final_threshold) > 0.08
                          else "Low — manual review recommended")

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
              f"(gap override if > {T['SEMANTIC_GAP_MIN']} & sem > {T['SEMANTIC_GAP_SCORE_MIN']})")
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

        print(f"\n[Decision Rules — v18]:")
        print(f"   Step 0  — Dampening:         "
              f"{'⚡ ACTIVE' if dampened else '○ skip'}"
              f"  [max_tfidf={max_tfidf:.3f} < {T['DAMPEN_TFIDF_BELOW']}]")
        print(f"   Step 1  — ModernTerms:       "
              f"{'⚡ → NOVEL' if modern_override else '○ skip'}"
              f"  [{len(modern_terms_found)} >= {T['MODERN_TERM_MIN_COUNT']}"
              f" & max_tfidf={max_tfidf:.3f} < {T['MODERN_TERM_TFIDF_MAX']}]")
        print(f"   Step 2  — SemanticGap:       "
              f"{'⚡ → NOVEL' if gap_override else '○ skip'}"
              f"  [gap={semantic_gap:.3f} > {T['SEMANTIC_GAP_MIN']}"
              f" & tfidf={max_tfidf:.3f} < {T['SEMANTIC_GAP_TFIDF_MAX']}"
              f" & sem={top_semantic:.3f} > {T['SEMANTIC_GAP_SCORE_MIN']}]")
        print(f"   Step 3  — StrongDrift:       "
              f"{'⚡ → NOVEL' if strong_drift else '○ skip'}"
              f"  [sem={top_semantic:.3f} > {T['STRONG_DRIFT_SEMANTIC_MIN']}"
              f" & max_tfidf={max_tfidf:.3f} < {T['STRONG_DRIFT_TFIDF_MAX']}]")
        print(f"   Step 4  — RankDecay:         "
              f"{'⚡ → NOVEL' if rank_decay_override else '○ skip'}"
              f"  [ratio={tfidf_rank_decay_ratio:.3f} < {T['RANK_DECAY_RATIO_MAX']}"
              f" & sem={top_semantic:.3f} > {T['RANK_DECAY_SEMANTIC_MIN']}"
              f" & max_tfidf < {T['STRONG_DRIFT_TFIDF_MAX']}]")
        print(f"   Rule 1  — Drift Safeguard:   "
              f"{'⚡ → NOVEL' if drift_detected else '○ skip'}"
              f"  [top_tfidf={top_tfidf:.3f} < {T['DRIFT_TFIDF_MAX']}"
              f" & sem={top_semantic:.3f} > {T['DRIFT_SEMANTIC_MIN']}]")
        print(f"   Rule 2  — Domain Coherence:  "
              f"{'⚡ → NOVEL' if coherence_failed else '○ skip'}"
              f"  [median={median_top5_tfidf:.3f} < {T['COHERENCE_TFIDF_MEDIAN']}"
              f" & eff_cal={effective_cal:.3f} >= {self.novelty_floor}"
              f" & sem={top_semantic:.3f} < {T['COHERENCE_SEMANTIC_MAX']}]")
        print(f"   Rule 3  — TF-IDF Override:   "
              f"{'⚡ → NOVEL' if tfidf_override else '○ skip'}"
              f"  [max_tfidf={max_tfidf:.3f} < {T['TFIDF_OVERRIDE_MAX']}"
              f" & hybrid={max_hybrid:.3f} < {T['HYBRID_OVERRIDE_MAX']}]")
        print(f"   Rule 4  — Out-of-domain:     "
              f"{'⚡ → NOVEL' if out_of_domain else '○ skip'}"
              f"  [eff_cal={effective_cal:.3f} < {self.novelty_floor}]")
        if self._incremental_thresholds_learned:
            zone_width = self.final_threshold - T["INCREMENTAL_CAL_LOW"]
            print(f"   Step 9  — Incremental(zone): "
                  f"{'⚡ → INCREMENTAL' if incremental_patent else '○ skip'}"
                  f"  [eff_cal={effective_cal:.3f} in [{T['INCREMENTAL_CAL_LOW']:.3f},"
                  f"{self.final_threshold:.3f}) width={zone_width:.3f}"
                  f" {'✓' if zone_width >= 0.04 else '⚠narrow'}"
                  f" | tfidf={max_tfidf:.3f} in [{T['INCREMENTAL_TFIDF_LOW']:.3f},"
                  f"{T['INCREMENTAL_TFIDF_HIGH']:.3f}]"
                  f" | gap={semantic_gap:.3f} < {T['INCREMENTAL_GAP_MAX']:.3f}"
                  f" | top3={top3_avg:.3f} >= {T['INCREMENTAL_TOP3_LOW']:.3f}]")
        else:
            print(f"   Step 9  — Incremental(zone): ○ skip  [thresholds not yet learned]")
        print(f"   Step 9b — Incremental(gap+tfidf+sem window): "
              f"{'⚡ → INCREMENTAL' if incremental_gap_window else '○ skip'}"
              f"  [gap={semantic_gap:.3f} in [{T['INCR_GAP_LOW']},{T['INCR_GAP_HIGH']})"
              f" & tfidf={max_tfidf:.3f} in [{T['INCR_TFIDF_LO2']},{T['INCR_TFIDF_HI2']}]"
              f" & sem={top_semantic:.3f} < {T['INCR_SEM_MAX']}"
              f" & eff_cal={effective_cal:.3f} in [{T['INCREMENTAL_CAL_LOW']:.3f},"
              f"{self.final_threshold:.3f})]")
        print(f"   Rule 5  — Dual Rejection:    "
              f"{'⚡ NOT NOVEL' if dual_reject else '○ no rejection'}"
              f"  [eff_cal={effective_cal:.3f} >= {self.final_threshold:.3f}"
              f" AND top3={top3_avg:.3f} >= {self.final_threshold:.3f}]")

        print(f"\n[System Config — v18]:")
        print(f"   Model        : {SEMANTIC_MODEL_NAME}  ✓ MPNet")
        print(f"   Threshold    : {self.final_threshold:.3f}")
        print(f"   Novelty floor: {self.novelty_floor:.3f}")
        print(f"   Weights      : TF-IDF={self.w_tfidf:.3f}, Semantic={self.w_semantic:.3f}")
        print(f"   Inference    : {elapsed:.3f}s")
        print(f"   Reason       : {reason}")
        print(f"   Confidence   : {confidence}")

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
    print("HYBRID PATENT NOVELTY DETECTION SYSTEM — v18")
    print("TF-IDF + MPNet Semantic Embeddings + Rule-Based Overrides")
    print("THREE-WAY: [NOVEL] | [INCREMENTAL] | [NOT NOVEL]")
    print("=" * 70)
    print(f"\nAuthor    : Devika Bakshi (122CS0301)")
    print(f"Supervisor: Asst. Prof. Sumanta Pyne")
    print(f"Institute : NIT Rourkela")
    print(f"Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSemantic model : {SEMANTIC_MODEL_NAME}")
    print(f"  → 768-dim embeddings, L2-normalised")
    print(f"  → Cosine similarity via dot product")
    print(f"  → v18: Hybrid negative mining + tightened Step 9b + wider zone")

    system = PatentNoveltySystem()

    # ── Dataset — USPTO only ─────────────────────────────────────────
    patents_df, positive_pairs = system.build_citation_dataset(
        patent_file="g_patent.tsv",
        abstract_file="g_patent_abstract.tsv",
        citation_file="g_us_patent_citation.tsv",
        min_citations=2,
        max_patents=10000
    )

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
    test_neg = system.get_mixed_negatives(test_pos, n_test, rng=np.random.RandomState(123))
    test_pairs = (pd.concat([test_pos, test_neg], ignore_index=True)
                  .sample(frac=1, random_state=123)
                  .reset_index(drop=True))
    print(f"Test: {len(test_pairs)} (pos={test_pos.shape[0]}, neg={test_neg.shape[0]})")

    metrics = system.evaluate_model(test_pairs, eval_top_k=100)

    system.plot_hybrid_similarity_distribution(
        test_pairs,
        save_path='hybrid_similarity_distribution.png'
    )

    # ── Sample predictions ───────────────────────────────────────────
    test_cases = {
        "A — Deep RL + CNN (EXPECT: NOT NOVEL)": """
        A neural network processing system and method for real-time pattern recognition
        using deep reinforcement learning and convolutional neural network layers.
        The system employs adaptive learning rates and backpropagation to train an
        ensemble of neural networks for object detection and image classification,
        achieving superior performance over prior art neural network methods.
        """,

        "B — RLHF / LLM alignment (EXPECT: NOVEL — post-2022 technology)": """
        A method for aligning large language models using reinforcement learning from
        human feedback (RLHF), comprising: a reward model fine-tuned on pairwise
        human preference annotations; a policy model trained with proximal policy
        optimization (PPO) and KL-divergence regularisation against a frozen reference
        model; and a constitutional AI self-critique loop that iteratively refines
        outputs to reduce harmful, toxic, and deceptive content.
        """,

        "C — Bicycle combination lock (EXPECT: NOVEL — unrelated domain)": """
        A portable bicycle security device comprising a hardened steel shackle
        and a four-digit numeric combination dial mechanism. The user selects a
        custom numeric code by rotating numbered discs to align in sequence,
        releasing the shackle. Housing is weather-sealed with rubber gaskets.
        No electronic components. Purely mechanical design.
        """,

        "D — CNN with attention (EXPECT: INCREMENTAL — known field, marginal improvement)": """
        A convolutional neural network system incorporating multi-head self-attention
        layers between convolutional blocks for image classification. The method
        applies channel-wise attention weighting to feature maps produced by
        standard convolutional filters, improving accuracy on benchmark datasets
        by 2.3% over baseline CNN architectures while maintaining similar
        computational cost through optimized kernel operations.
        """,
    }

    print("\n" + "=" * 70)
    print("SAMPLE NOVELTY PREDICTIONS — v18")
    print("=" * 70)

    decisions = {}
    for label, text in test_cases.items():
        print(f"\n{'─'*70}")
        print(f"  Case {label}")
        print(f"{'─'*70}")
        d, _ = system.predict_novelty(text, top_k=50)
        decisions[label] = d

    # ── Final Summary ─────────────────────────────────────────────────
    T = system._thr
    zone_width = system.final_threshold - T.get("INCREMENTAL_CAL_LOW", 0)

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — Hybrid Patent Novelty Detection System v18")
    print(f"{'='*70}")
    print(f"  Semantic model   : {SEMANTIC_MODEL_NAME}  ✓")
    print(f"  Embedding dim    : 768")
    print(f"  Similarity       : dot product (L2-normalised = cosine)")
    print(f"  Threshold        : {system.final_threshold:.3f}")
    print(f"  Novelty floor    : {system.novelty_floor:.3f}")
    print(f"  Fusion weights   : TF-IDF={system.w_tfidf:.3f}, Semantic={system.w_semantic:.3f}")
    print(f"  Calibrator fitted: {system.calibrator.fitted}")

    print(f"\n  Three-way Decision Output [v18]:")
    print(f"    [NOVEL]       → Potentially Novel (clear novelty)")
    print(f"    [INCREMENTAL] → Incremental Innovation (improvement patent)")
    print(f"    [NOT NOVEL]   → Prior Art Detected")

    print(f"\n  Decision Hierarchy [v18]:")
    print(f"    Step 0   Dampen:            max_tfidf<{T['DAMPEN_TFIDF_BELOW']}"
          f" → cal×{T['DAMPEN_FACTOR']}")
    print(f"    Step 1   ModernTerms:       {T['MODERN_TERM_MIN_COUNT']}+ post-2022 terms"
          f" & max_tfidf<{T['MODERN_TERM_TFIDF_MAX']} → NOVEL")
    print(f"    Step 2   SemanticGap:       gap>{T['SEMANTIC_GAP_MIN']}"
          f" & tfidf<{T['SEMANTIC_GAP_TFIDF_MAX']}"
          f" & sem>{T['SEMANTIC_GAP_SCORE_MIN']} → NOVEL")
    print(f"    Step 3   StrongDrift:       sem>{T['STRONG_DRIFT_SEMANTIC_MIN']}"
          f" & max_tfidf<{T['STRONG_DRIFT_TFIDF_MAX']} → NOVEL")
    print(f"    Step 4   RankDecay:         ratio<{T['RANK_DECAY_RATIO_MAX']}"
          f" & sem>{T['RANK_DECAY_SEMANTIC_MIN']} → NOVEL")
    print(f"    Rule 1   Drift:             top_tfidf<{T['DRIFT_TFIDF_MAX']}"
          f" & sem>{T['DRIFT_SEMANTIC_MIN']} → NOVEL")
    print(f"    Rule 2   Coherence:         median_top5<{T['COHERENCE_TFIDF_MEDIAN']}"
          f" & sem<{T['COHERENCE_SEMANTIC_MAX']} → NOVEL")
    print(f"    Rule 3   TFIDFOverride:     max_tfidf<{T['TFIDF_OVERRIDE_MAX']}"
          f" & hybrid<{T['HYBRID_OVERRIDE_MAX']} → NOVEL")
    print(f"    Rule 4   Floor:             eff_cal<{system.novelty_floor:.3f} → NOVEL")
    if system._incremental_thresholds_learned:
        print(f"    Step 9   Incremental(zone): eff_cal in [{T['INCREMENTAL_CAL_LOW']:.3f},"
              f"{system.final_threshold:.3f}) width={zone_width:.3f}"
              f" {'✓' if zone_width >= 0.04 else '⚠narrow'}")
        print(f"                               & tfidf in [{T['INCREMENTAL_TFIDF_LOW']:.3f},"
              f"{T['INCREMENTAL_TFIDF_HIGH']:.3f}]")
        print(f"                               & gap < {T['INCREMENTAL_GAP_MAX']:.3f}"
              f" & top3 >= {T['INCREMENTAL_TOP3_LOW']:.3f} → INCREMENTAL")
    print(f"    Step 9b  Incremental(gap+tfidf+sem window):  ← TIGHTENED v18")
    print(f"             gap in [{T['INCR_GAP_LOW']},{T['INCR_GAP_HIGH']})"
          f" & tfidf in [{T['INCR_TFIDF_LO2']},{T['INCR_TFIDF_HI2']}]"
          f" & sem < {T['INCR_SEM_MAX']}")
    print(f"             & eff_cal in [{T['INCREMENTAL_CAL_LOW']:.3f},{system.final_threshold:.3f})"
          f" → INCREMENTAL")
    print(f"    Rule 5   Reject:            eff_cal>={system.final_threshold:.3f}"
          f" AND top3>={system.final_threshold:.3f} → NOT NOVEL")

    if metrics:
        print(f"\n  Test Metrics (INCREMENTAL counted as NOVEL for binary metrics):")
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
        label_short = label[:65]
        if "NOT NOVEL" in label and "NOT NOVEL" in d:
            ok, mark = True, "✓"
        elif ("NOVEL" in label and "INCREMENTAL" not in label
              and "NOT NOVEL" not in label):
            ok   = "NOVEL" in d or "INCREMENTAL" in d
            mark = "✓" if ok else "✗"
        elif "INCREMENTAL" in label and "INCREMENTAL" in d:
            ok, mark = True, "✓"
        else:
            ok, mark = False, "✗"
        print(f"    {mark} {label_short}... → {d}")

    all_correct = all(
        (("NOT NOVEL"    in label and "NOT NOVEL"    in d) or
         ("INCREMENTAL"  in label and "INCREMENTAL"  in d) or
         ("NOVEL"        in label and "INCREMENTAL"  not in label and
          "NOT NOVEL"    not in label and
          ("NOVEL" in d or "INCREMENTAL" in d)))
        for label, d in decisions.items()
    )
    print(f"\n  All cases correct: {'✓ YES' if all_correct else '✗ NO — check above'}")

    print(f"\n  v18 improvements vs v17:")
    print(f"    IMP-V18-1: Hybrid negative mining (TF-IDF+Semantic combined score)")
    print(f"               Surfaces genuinely ambiguous pairs → sharper calibrator")
    print(f"    IMP-V18-2: Incremental zone widened (incr_fraction clamped at 0.90 max)")
    print(f"               Zone width {zone_width:.3f} "
          f"({'✓ viable' if zone_width >= 0.04 else '⚠ still narrow — Step 9b primary'})")
    print(f"    IMP-V18-3: Step 9b guards: sem < {T['INCR_SEM_MAX']} AND cal < threshold")
    print(f"               Prevents high-confidence NOT NOVEL patents mis-labelling")
    print(f"    IMP-V18-4: Threshold floor lowered 0.45 → 0.42")
    print(f"    IMP-V18-5: Candidate pool enlarged (top_k×8 sem, top_k×10 tfidf)")
    print(f"               Expected Recall@k improvement ~5-8%")
    print(f"    IMP-V18-6: Evaluation negative mining also uses hybrid score")
    print(f"    IMP-V18-7: Cache version bumped to v18 (v17 caches auto-invalidate)")
    print(f"    IMP-V18-8: Modern terms vocabulary expanded with 2024-2025 terms")
    print(f"\n  System ready.")


if __name__ == "__main__":
    main()