"""
Automated Patentability Check using NLP and Machine Learning
FINAL SEMESTER VERSION - April 2026

ENHANCEMENTS ADDED:
1. Random sampling for unbiased patent selection
2. BERT semantic similarity model
3. Hybrid TF-IDF + BERT similarity
4. Model comparison (TF-IDF vs BERT vs Hybrid)

Author: Devika Bakshi (122CS0301)
Supervisor: Asst. Prof. Sumanta Pyne
NIT Rourkela
"""

import pandas as pd
import numpy as np
import re
import json
from collections import Counter
from datetime import datetime

# NLTK for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-learn for TF-IDF and metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

# PyTorch and Transformers for BERT (Optional - CPU Compatible)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Note: Install torch and transformers for BERT semantic similarity")
    print("      Run: pip install torch transformers")

# Visualization
import matplotlib.pyplot as plt

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
if BERT_AVAILABLE:
    torch.manual_seed(42)


class PatentResearchSystem:
    """
    Automated Patentability Check System
    FINAL VERSION: TF-IDF + BERT + Hybrid Similarity
    """
    
    def __init__(self):
        """Initialize the patent research system."""
        # Text preprocessing components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # TF-IDF components
        self.vectorizer = None
        self.mid_threshold = None   # Boundary: ACCEPT vs INCREMENTAL
        self.high_threshold = None  # Boundary: INCREMENTAL vs REJECT
        
        # BERT components (Phase 2)
        self.tokenizer = None
        self.bert_model = None
        self.device = None
        self.bert_enabled = False
        
        # Storage
        self.test_pairs = None
        self.patents_df = None
        self.training_history = []
        
    # ============================================================
    # TEXT PREPROCESSING
    # ============================================================
    
    def preprocess(self, text):
        """
        Clean and preprocess patent text for TF-IDF.
        
        Steps:
        1. Convert to lowercase
        2. Remove non-alphabetic characters
        3. Tokenize
        4. Remove stopwords
        5. Lemmatize words
        """
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return " ".join(words)
    
    # ============================================================
    # DATASET CONSTRUCTION (WITH RANDOM SAMPLING)
    # ============================================================
    
    def build_citation_dataset(self, patent_file, abstract_file, citation_file,
                               min_citations=5, max_patents=10000, use_random_sampling=True):
        """
        Build citation-aware dataset with RANDOM SAMPLING (unbiased).
        
        ENHANCEMENT: Random sampling instead of sequential to avoid time-period bias.
        
        Args:
            patent_file: Path to g_patent.tsv
            abstract_file: Path to g_patent_abstract.tsv
            citation_file: Path to g_us_patent_citation.tsv
            min_citations: Minimum citations in both directions
            max_patents: Maximum number of patents to sample
            use_random_sampling: If True, use random sampling (recommended)
        """
        print("="*70)
        print("PHASE 1: BUILDING CITATION-AWARE DATASET")
        print("="*70)
        
        # --------------------------------------------------------
        # Step 1: Analyze citation graph
        # --------------------------------------------------------
        print("\n[1/5] Analyzing citation graph...")
        
        citing_counter = Counter()
        cited_counter = Counter()
        
        # Read first chunk to identify columns
        chunk_iter = pd.read_csv(citation_file, sep='\t', dtype=str, chunksize=500000)
        first_chunk = next(chunk_iter)
        cols = first_chunk.columns.tolist()
        print(f"Citation columns: {cols}")
        
        # Identify column names
        patent_col = None
        cited_col = None
        for col in cols:
            col_lower = col.lower()
            if 'citing' in col_lower or col_lower == 'patent_id':
                patent_col = col
            if 'cited' in col_lower or 'citation_patent_id' in col_lower:
                cited_col = col
        
        if patent_col is None or cited_col is None:
            # Fallback: use first two columns
            patent_col = cols[0]
            cited_col = cols[2] if len(cols) > 2 else cols[1]
            print(f"Using fallback columns: citing='{patent_col}', cited='{cited_col}'")
        
        print(f"Using columns: citing='{patent_col}', cited='{cited_col}'")
        
        # Process all chunks to count citations
        chunk_iter = pd.read_csv(citation_file, sep='\t', dtype=str,
                                 usecols=[patent_col, cited_col], chunksize=500000)
        
        for chunk in [first_chunk] + list(chunk_iter):
            citing_vals = chunk[patent_col].astype(str).fillna('').tolist()
            cited_vals = chunk[cited_col].astype(str).fillna('').tolist()
            
            for val in citing_vals:
                if val and val != 'nan':
                    citing_counter[val] += 1
            for val in cited_vals:
                if val and val != 'nan':
                    cited_counter[val] += 1
        
        # Identify core patents
        frequent_cited = {pid for pid, cnt in cited_counter.items() if cnt >= min_citations}
        frequent_citing = {pid for pid, cnt in citing_counter.items() if cnt >= min_citations}
        core_patents = list(frequent_cited.intersection(frequent_citing))
        total_core = len(core_patents)
        print(f"Found {total_core:,} core patents with bidirectional citations")
        
        # ========== ENHANCEMENT: RANDOM SAMPLING ==========
        if len(core_patents) > max_patents:
            if use_random_sampling:
                # RANDOM SAMPLING - Unbiased, matches report methodology
                core_patents = np.random.choice(core_patents, max_patents, replace=False).tolist()
                print(f"Randomly sampled {max_patents:,} core patents from {total_core:,}")
            else:
                # SEQUENTIAL SAMPLING - Biased, only for reproducibility
                core_patents = core_patents[:max_patents]
                print(f"Limited to first {max_patents:,} core patents (sequential)")
        # ==================================================
        
        # --------------------------------------------------------
        # Step 2: Load patent data
        # --------------------------------------------------------
        print("\n[2/5] Loading patent data...")
        
        patent_chunks = []
        for chunk in pd.read_csv(patent_file, sep='\t', dtype=str, chunksize=10000):
            if 'patent_id' in chunk.columns:
                mask = chunk['patent_id'].isin(core_patents)
                if mask.any():
                    patent_chunks.append(chunk[mask])
        
        if not patent_chunks:
            # Try to find patent_id column
            test_chunk = pd.read_csv(patent_file, sep='\t', nrows=5)
            pid_col = test_chunk.columns[0]
            for chunk in pd.read_csv(patent_file, sep='\t', dtype=str, chunksize=10000):
                mask = chunk[pid_col].isin(core_patents)
                if mask.any():
                    chunk = chunk.rename(columns={pid_col: 'patent_id'})
                    patent_chunks.append(chunk[mask])
        
        patents = pd.concat(patent_chunks, ignore_index=True)
        print(f"Loaded {len(patents):,} patents")
        
        # Load abstracts
        print("\n[3/5] Loading abstract data...")
        abstract_chunks = []
        patent_ids_set = set(patents['patent_id'].astype(str))
        
        for chunk in pd.read_csv(abstract_file, sep='\t', dtype=str, chunksize=10000):
            if 'patent_id' in chunk.columns:
                mask = chunk['patent_id'].astype(str).isin(patent_ids_set)
                if mask.any():
                    abstract_chunks.append(chunk[mask])
        
        abstracts = pd.concat(abstract_chunks, ignore_index=True) if abstract_chunks else pd.DataFrame()
        
        # Merge
        if len(abstracts) > 0:
            df = patents.merge(abstracts, on='patent_id', how='inner')
        else:
            df = patents.copy()
            df['patent_abstract'] = ""
        
        # Find title and abstract columns
        title_col = None
        abstract_col = None
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
        
        # Preprocess text
        print("\n[4/5] Preprocessing patent texts...")
        df['clean_text'] = (df['patent_title'] + " " + df['patent_abstract']).apply(self.preprocess)
        
        # --------------------------------------------------------
        # Step 3: Extract citation pairs
        # --------------------------------------------------------
        print("\n[5/5] Extracting citation pairs...")
        valid_ids = set(df['patent_id'].astype(str))
        
        citation_pairs = []
        chunk_iter = pd.read_csv(citation_file, sep='\t', dtype=str,
                                 usecols=[patent_col, cited_col], chunksize=500000)
        
        for chunk in chunk_iter:
            chunk = chunk.rename(columns={patent_col: 'patent_id', cited_col: 'cited_patent_id'})
            chunk['patent_id'] = chunk['patent_id'].astype(str)
            chunk['cited_patent_id'] = chunk['cited_patent_id'].astype(str)
            mask = chunk['patent_id'].isin(valid_ids) & chunk['cited_patent_id'].isin(valid_ids)
            if mask.any():
                citation_pairs.append(chunk[mask])
        
        citations = pd.concat(citation_pairs, ignore_index=True) if citation_pairs else pd.DataFrame()
        print(f"Found {len(citations):,} citation pairs")
        
        if len(citations) == 0:
            print("No citation pairs found. Using demo data.")
            return self._create_demo_data()
        
        # --------------------------------------------------------
        # Step 4: Create training pairs
        # --------------------------------------------------------
        print("\nCreating balanced training pairs...")
        
        # Positive pairs (citations)
        n_positive = min(2000, len(citations))
        positive_pairs = citations.sample(n=n_positive, random_state=42).copy()
        positive_pairs['label'] = 1
        
        # Negative pairs (random non-citations)
        all_patents = list(valid_ids)
        citation_set = set(zip(citations['patent_id'], citations['cited_patent_id']))
        
        negative_pairs = []
        while len(negative_pairs) < n_positive:
            p1 = np.random.choice(all_patents)
            p2 = np.random.choice(all_patents)
            if p1 != p2 and (p1, p2) not in citation_set and (p2, p1) not in citation_set:
                negative_pairs.append({'patent_id': p1, 'cited_patent_id': p2})
        
        negative_pairs = pd.DataFrame(negative_pairs)
        negative_pairs['label'] = 0
        
        all_pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        
        # Add text
        patent_texts = df.set_index('patent_id')['clean_text'].to_dict()
        all_pairs['text_1'] = all_pairs['patent_id'].map(patent_texts)
        all_pairs['text_2'] = all_pairs['cited_patent_id'].map(patent_texts)
        all_pairs = all_pairs.dropna()
        
        print(f"Created {len(all_pairs):,} training pairs")
        print(f"  Positive: {len(positive_pairs):,}")
        print(f"  Negative: {len(negative_pairs):,}")
        
        self.patents_df = df
        return df, all_pairs
    
    # ============================================================
    # DEMO DATA (Fallback)
    # ============================================================
    
    def _create_demo_data(self, num_patents=100):
        """Create synthetic demo data for testing."""
        print(f"\nCreating demo dataset with {num_patents} patents...")
        
        domains = {
            'neural_networks': ['neural network', 'deep learning', 'backpropagation', 'LSTM', 'transformer'],
            'computer_vision': ['object detection', 'image segmentation', 'face recognition', 'convolution'],
            'nlp': ['text classification', 'sentiment analysis', 'machine translation', 'BERT', 'embedding'],
            'reinforcement_learning': ['Q-learning', 'policy gradient', 'deep Q network', 'actor-critic'],
            'optimization': ['gradient descent', 'Adam optimizer', 'learning rate', 'regularization']
        }
        
        patents = []
        pid = 1
        
        for domain, terms in domains.items():
            for i in range(num_patents // len(domains)):
                main_term = np.random.choice(terms)
                title = f"System for {main_term.title()}"
                selected_terms = np.random.choice(terms, size=np.random.randint(3, 5), replace=False)
                abstract = f"A novel {domain.replace('_', ' ')} method. Implements {', '.join(selected_terms)}."
                patents.append({
                    'patent_id': f"PAT{pid:04d}",
                    'patent_title': title,
                    'patent_abstract': abstract
                })
                pid += 1
        
        patents_df = pd.DataFrame(patents)
        patents_df['clean_text'] = (patents_df['patent_title'] + " " + patents_df['patent_abstract']).apply(self.preprocess)
        
        # Create citation pairs
        citation_pairs = []
        for i, row1 in patents_df.iterrows():
            for j, row2 in patents_df.iterrows():
                if i >= j:
                    continue
                words1 = set(row1['clean_text'].split())
                words2 = set(row2['clean_text'].split())
                overlap = len(words1.intersection(words2))
                if overlap > 2 and np.random.random() > 0.85:
                    citation_pairs.append((row1['patent_id'], row2['patent_id']))
                if len(citation_pairs) >= num_patents:
                    break
            if len(citation_pairs) >= num_patents:
                break
        
        citations = pd.DataFrame(citation_pairs, columns=['patent_id', 'cited_patent_id'])
        citations['label'] = 1
        
        # Negative pairs
        all_patents = patents_df['patent_id'].tolist()
        negative_pairs = []
        citation_set = set(zip(citations['patent_id'], citations['cited_patent_id']))
        
        while len(negative_pairs) < len(citations):
            p1 = np.random.choice(all_patents)
            p2 = np.random.choice(all_patents)
            if p1 != p2 and (p1, p2) not in citation_set:
                negative_pairs.append({'patent_id': p1, 'cited_patent_id': p2})
        
        negatives = pd.DataFrame(negative_pairs)
        negatives['label'] = 0
        
        all_pairs = pd.concat([citations, negatives], ignore_index=True)
        
        patent_texts = patents_df.set_index('patent_id')['clean_text'].to_dict()
        all_pairs['text_1'] = all_pairs['patent_id'].map(patent_texts)
        all_pairs['text_2'] = all_pairs['cited_patent_id'].map(patent_texts)
        all_pairs = all_pairs.dropna()
        
        print(f"Demo data created: {len(patents_df)} patents, {len(all_pairs)} pairs")
        self.patents_df = patents_df
        return patents_df, all_pairs
    
    # ============================================================
    # TF-IDF TRAINING & THRESHOLD LEARNING
    # ============================================================
    
    def train_and_evaluate(self, pairs_df):
        """Train TF-IDF model and learn classification thresholds."""
        print("\n" + "="*70)
        print("TF-IDF MODEL TRAINING & THRESHOLD LEARNING")
        print("="*70)
        
        # Split data
        train_pairs, temp_pairs = train_test_split(
            pairs_df, test_size=0.3, random_state=42, stratify=pairs_df['label']
        )
        val_pairs, test_pairs = train_test_split(
            temp_pairs, test_size=0.5, random_state=42, stratify=temp_pairs['label']
        )
        print(f"\nData Split:")
        print(f"  Training: {len(train_pairs):,} pairs")
        print(f"  Validation: {len(val_pairs):,} pairs")
        print(f"  Test: {len(test_pairs):,} pairs")
        
        # Fit TF-IDF
        all_texts = pd.concat([pairs_df['text_1'], pairs_df['text_2']]).unique()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9
        )
        self.vectorizer.fit(all_texts)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        
        # Compute similarities
        def compute_similarity(row):
            v1 = self.vectorizer.transform([row['text_1']])
            v2 = self.vectorizer.transform([row['text_2']])
            return cosine_similarity(v1, v2)[0][0]
        
        print("\nComputing cosine similarities...")
        train_pairs['similarity'] = train_pairs.apply(compute_similarity, axis=1)
        val_pairs['similarity'] = val_pairs.apply(compute_similarity, axis=1)
        test_pairs['similarity'] = test_pairs.apply(compute_similarity, axis=1)
        
        # Learn mid_threshold
        thresholds = np.arange(0.01, 0.5, 0.01)
        best_f1 = 0
        best_thresh = 0.05
        
        for thresh in thresholds:
            pred = (val_pairs['similarity'] >= thresh).astype(int)
            if val_pairs['label'].sum() > 0:
                f1 = f1_score(val_pairs['label'], pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        
        self.mid_threshold = best_thresh
        print(f"\nLearned MID threshold: {self.mid_threshold:.3f} (F1={best_f1:.3f})")
        
        # Learn high_threshold
        pos_sims = val_pairs[val_pairs['label'] == 1]['similarity']
        if len(pos_sims) > 0:
            self.high_threshold = np.percentile(pos_sims, 75)
            print(f"Learned HIGH threshold (75th percentile): {self.high_threshold:.3f}")
        else:
            self.high_threshold = self.mid_threshold * 2
        
        # Evaluate
        test_pred = (test_pairs['similarity'] >= self.mid_threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(test_pairs['label'], test_pred),
            'precision': precision_score(test_pairs['label'], test_pred, zero_division=0),
            'recall': recall_score(test_pairs['label'], test_pred, zero_division=0),
            'f1': f1_score(test_pairs['label'], test_pred, zero_division=0),
            'auc': roc_auc_score(test_pairs['label'], test_pairs['similarity']),
            'mid_threshold': self.mid_threshold,
            'high_threshold': self.high_threshold,
            'confusion_matrix': confusion_matrix(test_pairs['label'], test_pred).tolist()
        }
        
        print("\n" + "-"*50)
        print("TEST SET PERFORMANCE (TF-IDF)")
        print("-"*50)
        print(f"Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1-Score:  {metrics['f1']:.3f}")
        print(f"AUC-ROC:   {metrics['auc']:.3f}")
        
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0][0]}  FP: {cm[0][1]}")
        print(f"  FN: {cm[1][0]}  TP: {cm[1][1]}")
        
        self.test_pairs = test_pairs
        self._plot_tfidf_results(test_pairs, metrics['auc'])
        
        return metrics
    
    def _plot_tfidf_results(self, test_pairs, auc):
        """Plot TF-IDF evaluation results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(test_pairs[test_pairs['label']==1]['similarity'], 
                     bins=30, alpha=0.7, label='Citation Pairs', color='green')
        axes[0].hist(test_pairs[test_pairs['label']==0]['similarity'], 
                     bins=30, alpha=0.7, label='Random Pairs', color='red')
        axes[0].axvline(self.mid_threshold, color='blue', linestyle='--',
                        label=f'Mid Threshold={self.mid_threshold:.2f}')
        axes[0].axvline(self.high_threshold, color='purple', linestyle=':',
                        label=f'High Threshold={self.high_threshold:.2f}')
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Similarity Distribution')
        axes[0].legend()
        
        fpr, tpr, _ = roc_curve(test_pairs['label'], test_pairs['similarity'])
        axes[1].plot(fpr, tpr, 'b-', label=f'AUC = {auc:.3f}')
        axes[1].plot([0, 1], [0, 1], 'r--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        
        prec_vals, rec_vals, _ = precision_recall_curve(test_pairs['label'], 
                                                         test_pairs['similarity'])
        axes[2].plot(rec_vals, prec_vals, 'g-')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curve')
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('tfidf_evaluation_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nPlot saved as 'tfidf_evaluation_results.png'")
    
    # ============================================================
    # PHASE 2: BERT SEMANTIC MODEL (NEW)
    # ============================================================
    
    def init_bert(self, model_name="bert-base-uncased"):
        """Initialize BERT model for semantic similarity."""
        if not BERT_AVAILABLE:
            print("\n⚠️ BERT not available. Install with: pip install torch transformers")
            return False
        
        print("\n" + "="*70)
        print("INITIALIZING BERT SEMANTIC MODEL")
        print("="*70)
        
        try:
            print(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            
            # Use CPU (as requested)
            self.device = torch.device('cpu')
            self.bert_model.to(self.device)
            self.bert_model.eval()
            self.bert_enabled = True
            
            print(f"✓ BERT loaded successfully on CPU")
            return True
        except Exception as e:
            print(f"Failed to load BERT: {e}")
            return False
    
    def get_bert_embedding(self, text, max_length=512):
        """Get BERT embedding for text."""
        if not self.bert_enabled:
            return None
        
        tokens = self.tokenizer(text, max_length=max_length, padding=True, 
                                truncation=True, return_tensors='pt')
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**tokens)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    def compute_bert_similarity(self, text1, text2):
        """Compute BERT-based cosine similarity."""
        emb1 = self.get_bert_embedding(text1)
        emb2 = self.get_bert_embedding(text2)
        if emb1 is None or emb2 is None:
            return None
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def compute_hybrid_similarity(self, text1, text2, alpha=0.5):
        """
        Combine TF-IDF and BERT similarities.
        
        Args:
            alpha: Weight for TF-IDF (1-alpha for BERT)
        """
        clean1 = self.preprocess(text1)
        clean2 = self.preprocess(text2)
        
        # TF-IDF similarity
        v1 = self.vectorizer.transform([clean1])
        v2 = self.vectorizer.transform([clean2])
        tfidf_sim = cosine_similarity(v1, v2)[0][0]
        
        # BERT similarity
        if self.bert_enabled:
            bert_sim = self.compute_bert_similarity(text1, text2)
        else:
            bert_sim = tfidf_sim
        
        # Hybrid
        hybrid_sim = alpha * tfidf_sim + (1 - alpha) * bert_sim
        
        return {
            'tfidf': tfidf_sim,
            'bert': bert_sim,
            'hybrid': hybrid_sim,
            'alpha': alpha
        }
    
    # ============================================================
    # NOVELTY PREDICTION (Enhanced)
    # ============================================================
    
    def predict_novelty(self, new_text, method='tfidf', alpha=0.5):
        """
        Predict novelty of a new patent text.
        
        Args:
            new_text: Patent description text
            method: 'tfidf', 'bert', or 'hybrid'
            alpha: Weight for TF-IDF in hybrid method
        """
        if self.patents_df is None:
            raise ValueError("No patent data loaded. Run build_citation_dataset first.")
        
        print("\n" + "="*70)
        print(f"NOVELTY PREDICTION (Method: {method.upper()})")
        print("="*70)
        
        print(f"\nQuery Patent:")
        print(f"  {new_text[:200]}...")
        
        clean_new = self.preprocess(new_text)
        results = []
        
        print(f"\nComparing with {len(self.patents_df):,} existing patents...")
        
        for _, row in self.patents_df.iterrows():
            if method == 'tfidf':
                vec_new = self.vectorizer.transform([clean_new])
                vec_existing = self.vectorizer.transform([row['clean_text']])
                sim = cosine_similarity(vec_new, vec_existing)[0][0]
            elif method == 'bert' and self.bert_enabled:
                sim = self.compute_bert_similarity(clean_new, row['clean_text'])
            elif method == 'hybrid' and self.bert_enabled:
                sims = self.compute_hybrid_similarity(clean_new, row['clean_text'], alpha)
                sim = sims['hybrid']
            else:
                # Fallback to TF-IDF
                vec_new = self.vectorizer.transform([clean_new])
                vec_existing = self.vectorizer.transform([row['clean_text']])
                sim = cosine_similarity(vec_new, vec_existing)[0][0]
            
            results.append({
                'patent_id': row['patent_id'],
                'title': row['patent_title'],
                'similarity': sim
            })
        
        results_df = pd.DataFrame(results)
        max_sim = results_df['similarity'].max()
        
        print(f"\nMax Similarity: {max_sim:.4f}")
        print(f"Mean Similarity: {results_df['similarity'].mean():.4f}")
        
        # Top 5 similar patents
        top5 = results_df.nlargest(5, 'similarity')
        print("\n" + "-"*50)
        print("TOP 5 SIMILAR PATENTS")
        print("-"*50)
        for _, row in top5.iterrows():
            print(f"\n  Patent: {row['patent_id']}")
            print(f"  Title: {row['title'][:80]}...")
            print(f"    Similarity: {row['similarity']:.4f}")
        
        # Decision
        if max_sim >= self.high_threshold:
            decision = "REJECT (Highly Similar - Not Patentable)"
        elif max_sim >= self.mid_threshold:
            decision = "INCREMENTAL (Builds on Existing Work)"
        else:
            decision = "ACCEPT (Potentially Novel)"
        
        print("\n" + "="*50)
        print(f"DECISION: {decision}")
        print("="*50)
        
        return decision, results_df
    
    # ============================================================
    # MODEL COMPARISON (NEW)
    # ============================================================
    
    def compare_models(self, alpha=0.5):
        """Compare TF-IDF, BERT, and Hybrid models."""
        if self.test_pairs is None:
            print("No test data available. Run train_and_evaluate first.")
            return None
        
        print("\n" + "="*70)
        print("MODEL COMPARISON: TF-IDF vs BERT vs HYBRID")
        print("="*70)
        
        results = []
        print("\nComputing similarities for all models...")
        
        for idx, row in self.test_pairs.iterrows():
            if idx % 50 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(self.test_pairs)}")
            
            # TF-IDF
            v1 = self.vectorizer.transform([row['text_1']])
            v2 = self.vectorizer.transform([row['text_2']])
            tfidf_sim = cosine_similarity(v1, v2)[0][0]
            
            # BERT (if available)
            if self.bert_enabled:
                bert_sim = self.compute_bert_similarity(row['text_1'], row['text_2'])
            else:
                bert_sim = tfidf_sim
            
            # Hybrid
            hybrid_sim = alpha * tfidf_sim + (1 - alpha) * bert_sim
            
            results.append({
                'label': row['label'],
                'tfidf_sim': tfidf_sim,
                'bert_sim': bert_sim,
                'hybrid_sim': hybrid_sim
            })
        
        eval_df = pd.DataFrame(results)
        
        # Compute metrics
        metrics = {}
        for model_name, sim_col in [('TF-IDF', 'tfidf_sim'), 
                                     ('BERT', 'bert_sim'), 
                                     ('Hybrid', 'hybrid_sim')]:
            pred = (eval_df[sim_col] >= self.mid_threshold).astype(int)
            metrics[model_name] = {
                'accuracy': accuracy_score(eval_df['label'], pred),
                'precision': precision_score(eval_df['label'], pred, zero_division=0),
                'recall': recall_score(eval_df['label'], pred, zero_division=0),
                'f1': f1_score(eval_df['label'], pred, zero_division=0),
                'auc': roc_auc_score(eval_df['label'], eval_df[sim_col])
            }
        
        # Print results
        print("\n" + "-"*70)
        print("MODEL COMPARISON RESULTS")
        print("-"*70)
        print(f"{'Model':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC-ROC':<12}")
        print("-"*70)
        
        for model_name, m in metrics.items():
            print(f"{model_name:<12} {m['accuracy']:<12.3f} {m['precision']:<12.3f} "
                  f"{m['recall']:<12.3f} {m['f1']:<12.3f} {m['auc']:<12.3f}")
        
        self._plot_comparison(metrics)
        return metrics
    
    def _plot_comparison(self, metrics):
        """Plot model comparison bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(metric_names))
        width = 0.25
        
        for i, model in enumerate(models):
            values = [metrics[model][m] for m in metric_names]
            axes[0].bar(x + i*width, values, width, label=model)
        
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(metric_names)
        axes[0].legend()
        axes[0].set_ylim([0, 1])
        
        auc_values = [metrics[model]['auc'] for model in models]
        bars = axes[1].bar(models, auc_values, color=['blue', 'green', 'orange'])
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_title('AUC-ROC Comparison')
        axes[1].set_ylim([0, 1])
        
        for bar, val in zip(bars, auc_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nPlot saved as 'model_comparison.png'")
    
    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================
    
    def save_results(self, filename='patent_final_results.json'):
        """Save all results to JSON file."""
        results = {
            'model_info': {
                'mid_threshold': float(self.mid_threshold) if self.mid_threshold else None,
                'high_threshold': float(self.high_threshold) if self.high_threshold else None,
                'vocabulary_size': len(self.vectorizer.vocabulary_) if self.vectorizer else None,
                'bert_enabled': self.bert_enabled
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to '{filename}'")


# ============================================================
# MAIN FUNCTIONS
# ============================================================

def main():
    """Main execution function for final semester."""
    print("="*70)
    print("AUTOMATED PATENTABILITY CHECK SYSTEM - FINAL VERSION")
    print("Enhancements: Random Sampling + BERT + Hybrid Similarity")
    print("="*70)
    print(f"\nAuthor: Devika Bakshi (122CS0301)")
    print(f"Supervisor: Asst. Prof. Sumanta Pyne")
    print(f"NIT Rourkela")
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    system = PatentResearchSystem()
    
    # Build dataset with RANDOM SAMPLING (key enhancement)
    patent_file = "g_patent.tsv"
    abstract_file = "g_patent_abstract.tsv"
    citation_file = "g_us_patent_citation.tsv"
    
    try:
        patents_df, pairs_df = system.build_citation_dataset(
            patent_file=patent_file,
            abstract_file=abstract_file,
            citation_file=citation_file,
            min_citations=5,
            max_patents=10000,
            use_random_sampling=True  # ← ENHANCEMENT: Random sampling
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nUsing demo data: {e}")
        patents_df, pairs_df = system._create_demo_data(num_patents=200)
    
    # Train TF-IDF model
    tfidf_metrics = system.train_and_evaluate(pairs_df)
    
    # Optional: Try BERT
    try_bert = input("\n\nEnable BERT semantic model? (y/n - requires torch/transformers): ").lower() == 'y'
    
    if try_bert:
        if system.init_bert():
            # Compare all models
            print("\n" + "="*70)
            print("RUNNING MODEL COMPARISON")
            print("="*70)
            alpha = 0.5
            comparison = system.compare_models(alpha=alpha)
            
            # Hybrid prediction
            test_patent = """
            A novel quantum-inspired neural network architecture for real-time
            pattern recognition in high-dimensional data spaces. The system
            combines adaptive resonance theory with deep reinforcement learning
            to achieve unprecedented accuracy in streaming data scenarios.
            """
            
            decision, _ = system.predict_novelty(test_patent, method='hybrid', alpha=0.5)
    
    # Save results
    system.save_results()
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def quick_test():
    """Quick test with demo data."""
    print("="*70)
    print("QUICK TEST - Final Version")
    print("="*70)
    
    system = PatentResearchSystem()
    patents_df, pairs_df = system._create_demo_data(num_patents=100)
    metrics = system.train_and_evaluate(pairs_df)
    
    test_patent = "A quantum-inspired neural network for pattern recognition"
    decision, _ = system.predict_novelty(test_patent, method='tfidf')
    
    system.save_results('quick_test_results.json')
    print("\nQuick test completed!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_test()
    else:
        main()