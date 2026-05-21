"""
Automated Novelty Check System for Patent Pre-Screening
NLP + Machine Learning for Prior Art Similarity Assessment

Author: Devika Bakshi (122CS0301)
Supervisor: Asst. Prof. Sumanta Pyne
NIT Rourkela
"""

import pandas as pd
import numpy as np
import re
import time
import pickle
import os
import warnings
import logging
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

# PyTorch and Sentence-BERT
import torch
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Note: Install sentence-transformers for semantic similarity")

# Visualization
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
if SBERT_AVAILABLE:
    torch.manual_seed(42)


class PatentNoveltySystem:
    
    def __init__(self, model_dir='models/'):
        """Initialize the patent novelty system."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        self.vectorizer = None
        self.thresholds = {}
        
        self.sbert_model = None
        self.sbert_enabled = False
        self.patent_embeddings = None
        self.patent_ids_ordered = None
        self.id_to_index = None
        
        self.text_map = None
        self.title_map = None
        self.tfidf_matrix = None
        self.tfidf_cache = None
        
        self.test_pairs = None
        self.patents_df = None
        self.model_dir = model_dir
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.inference_times = {}
        self.tfidf_min = None
        self.tfidf_max = None
        
    # ============================================================
    # TEXT PREPROCESSING
    # ============================================================
    
    def preprocess(self, text):
        if pd.isna(text) or not str(text).strip():
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return " ".join(words)
    
    def jaccard_similarity(self, text1, text2):
        s1 = set(text1.split())
        s2 = set(text2.split())
        if len(s1 | s2) == 0:
            return 0.0
        return len(s1 & s2) / (len(s1 | s2) + 1e-8)
    
    # ============================================================
    # DATASET CONSTRUCTION
    # ============================================================
    
    def build_citation_dataset(self, patent_file, abstract_file, citation_file,
                               min_citations=5, max_patents=10000, 
                               use_random_sampling=True):
        print("="*70)
        print("BUILDING CITATION-AWARE DATASET")
        print("="*70)
        
        print("\n[1/6] Analyzing citation graph...")
        
        citing_counter = Counter()
        cited_counter = Counter()
        
        chunk_iter = pd.read_csv(citation_file, sep='\t', dtype=str, chunksize=500000)
        first_chunk = next(chunk_iter)
        cols = first_chunk.columns.tolist()
        
        patent_col = None
        cited_col = None
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
        
        frequent_cited = {pid for pid, cnt in cited_counter.items() if cnt >= min_citations}
        frequent_citing = {pid for pid, cnt in citing_counter.items() if cnt >= min_citations}
        core_patents = list(frequent_cited.intersection(frequent_citing))
        print(f"Found {len(core_patents):,} core patents")
        
        if len(core_patents) > max_patents:
            if use_random_sampling:
                core_patents = np.random.choice(core_patents, max_patents, replace=False).tolist()
                print(f"Randomly sampled {max_patents:,} core patents")
            else:
                core_patents = core_patents[:max_patents]
        
        print("\n[2/6] Loading patent data...")
        patent_chunks = []
        for chunk in pd.read_csv(patent_file, sep='\t', dtype=str, chunksize=10000):
            if 'patent_id' in chunk.columns:
                mask = chunk['patent_id'].isin(core_patents)
                if mask.any():
                    patent_chunks.append(chunk[mask])
        
        patents = pd.concat(patent_chunks, ignore_index=True)
        print(f"Loaded {len(patents):,} patents")
        
        print("\n[3/6] Loading abstract data...")
        abstract_chunks = []
        patent_ids_set = set(patents['patent_id'].astype(str))
        
        for chunk in pd.read_csv(abstract_file, sep='\t', dtype=str, chunksize=10000):
            if 'patent_id' in chunk.columns:
                mask = chunk['patent_id'].astype(str).isin(patent_ids_set)
                if mask.any():
                    abstract_chunks.append(chunk[mask])
        
        abstracts = pd.concat(abstract_chunks, ignore_index=True) if abstract_chunks else pd.DataFrame()
        
        if len(abstracts) > 0:
            df = patents.merge(abstracts, on='patent_id', how='inner')
        else:
            df = patents.copy()
            df['patent_abstract'] = ""
        
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
        
        print("\n[4/6] Preprocessing patent texts...")
        df['clean_text'] = (df['patent_title'] + " " + df['patent_abstract']).apply(self.preprocess)
        
        self.text_map = dict(zip(df['patent_id'], df['clean_text']))
        self.title_map = dict(zip(df['patent_id'], df['patent_title']))
        
        print("\n[5/6] Extracting citation pairs...")
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
        print(f"Raw citation pairs: {len(citations):,}")
        
        if len(citations) == 0:
            return self._create_demo_data()
        
        # ============================================================
        # STEP 6: FIXED - ROBUST PAIR CREATION
        # ============================================================
        print("\n[6/6] Creating training pairs (balanced)...")
        
        # Positive pairs
        n_positive = min(1000, len(citations))
        positive_pairs = citations.sample(n=n_positive, random_state=42).copy()
        positive_pairs['label'] = 1
        
        # Negative pairs (robust with max_attempts)
        all_patents = list(valid_ids)
        citation_set = set(zip(citations['patent_id'], citations['cited_patent_id']))
        
        negative_pairs = []
        max_attempts = 50000
        attempts = 0
        
        while len(negative_pairs) < n_positive and attempts < max_attempts:
            p1 = np.random.choice(all_patents)
            p2 = np.random.choice(all_patents)
            
            if p1 != p2 and (p1, p2) not in citation_set and (p2, p1) not in citation_set:
                negative_pairs.append({
                    'patent_id': p1,
                    'cited_patent_id': p2,
                    'label': 0
                })
            attempts += 1
        
        negative_pairs = pd.DataFrame(negative_pairs)
        
        # Combine
        all_pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        
        # Attach text
        all_pairs['text_1'] = all_pairs['patent_id'].map(self.text_map)
        all_pairs['text_2'] = all_pairs['cited_patent_id'].map(self.text_map)
        
        # Drop missing safely
        all_pairs = all_pairs.dropna()
        
        # FINAL SAFETY CHECK
        print("\nLabel distribution:")
        print(all_pairs['label'].value_counts())
        
        if all_pairs['label'].nunique() < 2:
            raise ValueError("Dataset still imbalanced — check data loading.")
        
        print(f"\nCreated {len(all_pairs)} training pairs")
        print(f"  Positive: {len(positive_pairs)}")
        print(f"  Negative: {len(negative_pairs)}")
        
        self.patents_df = df
        return df, all_pairs
    
    def _create_demo_data(self, num_patents=200):
        """Create synthetic demo data."""
        print(f"\nCreating demo dataset with {num_patents} patents...")
        
        domains = {
            'neural_networks': ['neural network', 'deep learning', 'backpropagation', 'LSTM', 'transformer'],
            'computer_vision': ['object detection', 'image segmentation', 'face recognition', 'convolution'],
            'nlp': ['text classification', 'sentiment analysis', 'machine translation', 'BERT'],
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
        
        citations = pd.DataFrame(citation_pairs, columns=['patent_id', 'cited_patent_id'])
        citations['label'] = 1
        
        all_patents = patents_df['patent_id'].tolist()
        citation_set = set(zip(citations['patent_id'], citations['cited_patent_id']))
        
        negative_pairs = []
        while len(negative_pairs) < len(citations):
            p1 = np.random.choice(all_patents)
            p2 = np.random.choice(all_patents)
            if p1 != p2 and (p1, p2) not in citation_set:
                negative_pairs.append({'patent_id': p1, 'cited_patent_id': p2, 'label': 0})
        
        negatives = pd.DataFrame(negative_pairs[:len(citations)])
        all_pairs = pd.concat([citations, negatives], ignore_index=True)
        
        all_pairs['text_1'] = all_pairs['patent_id'].map(dict(zip(patents_df['patent_id'], patents_df['clean_text'])))
        all_pairs['text_2'] = all_pairs['cited_patent_id'].map(dict(zip(patents_df['patent_id'], patents_df['clean_text'])))
        all_pairs = all_pairs.dropna()
        
        print(f"Demo data created: {len(patents_df)} patents, {len(all_pairs)} pairs")
        
        self.text_map = dict(zip(patents_df['patent_id'], patents_df['clean_text']))
        self.title_map = dict(zip(patents_df['patent_id'], patents_df['patent_title']))
        self.patents_df = patents_df
        return patents_df, all_pairs
    
    # ============================================================
    # TF-IDF TRAINING
    # ============================================================
    
    def train_tfidf(self, pairs_df):
        print("\n" + "="*70)
        print("TF-IDF MODEL TRAINING")
        print("="*70)
        
        if pairs_df['label'].nunique() < 2:
            logger.error(f"Only {pairs_df['label'].nunique()} class(es) found. Cannot train.")
            logger.error(f"Label distribution: {pairs_df['label'].value_counts().to_dict()}")
            return None
        
        try:
            train_pairs, temp_pairs = train_test_split(
                pairs_df, test_size=0.3, random_state=42, stratify=pairs_df['label']
            )
            val_pairs, test_pairs = train_test_split(
                temp_pairs, test_size=0.5, random_state=42, stratify=temp_pairs['label']
            )
        except ValueError as e:
            logger.error(f"Train-test split failed: {e}")
            return None
        
        print(f"\nData Split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
        
        all_texts = pd.concat([pairs_df['text_1'], pairs_df['text_2']]).unique()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=2, max_df=0.9)
        self.vectorizer.fit(all_texts)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        
        self.patent_ids_ordered = self.patents_df['patent_id'].tolist()
        tfidf_matrix_list = []
        for pid in self.patent_ids_ordered:
            text = self.text_map[pid]
            tfidf_matrix_list.append(self.vectorizer.transform([text]))
        
        from scipy.sparse import vstack
        self.tfidf_matrix = vstack(tfidf_matrix_list)
        self.tfidf_cache = {pid: vec for pid, vec in zip(self.patent_ids_ordered, tfidf_matrix_list)}
        
        def compute_sim(row):
            v1 = self.tfidf_cache[row['patent_id']]
            v2 = self.tfidf_cache[row['cited_patent_id']]
            return cosine_similarity(v1, v2)[0][0]
        
        print("\nComputing TF-IDF similarities...")
        train_pairs['similarity'] = train_pairs.apply(compute_sim, axis=1)
        val_pairs['similarity'] = val_pairs.apply(compute_sim, axis=1)
        test_pairs['similarity'] = test_pairs.apply(compute_sim, axis=1)
        
        all_sims = pd.concat([train_pairs['similarity'], val_pairs['similarity'], test_pairs['similarity']])
        self.tfidf_min = all_sims.min()
        self.tfidf_max = all_sims.max()
        
        tfidf_thresh = self._optimize_threshold_pr(val_pairs['similarity'], val_pairs['label'])
        self.thresholds['TF-IDF'] = tfidf_thresh
        
        pos_sims = val_pairs[val_pairs['label'] == 1]['similarity']
        self.high_threshold = np.percentile(pos_sims, 75) if len(pos_sims) > 0 else tfidf_thresh * 2
        
        print(f"\nLearned thresholds:")
        print(f"  TF-IDF: {tfidf_thresh:.3f}")
        print(f"  REJECT: {self.high_threshold:.3f}")
        
        test_pred = (test_pairs['similarity'] >= tfidf_thresh).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(test_pairs['label'], test_pred),
            'precision': precision_score(test_pairs['label'], test_pred, zero_division=0),
            'recall': recall_score(test_pairs['label'], test_pred, zero_division=0),
            'f1': f1_score(test_pairs['label'], test_pred, zero_division=0),
            'auc': roc_auc_score(test_pairs['label'], test_pairs['similarity']),
            'threshold': tfidf_thresh,
            'high_threshold': self.high_threshold
        }
        
        print("\n" + "="*70)
        print("FINAL MODEL PERFORMANCE")
        print("="*70)
        print(f"Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1']:.3f}")
        print(f"AUC-ROC:   {metrics['auc']:.3f}")
        print("="*70)
        
        self.test_pairs = test_pairs
        self._plot_results(test_pairs, metrics['auc'])
        
        return metrics
    
    def _optimize_threshold_pr(self, similarities, labels):
        precision, recall, thresholds = precision_recall_curve(labels, similarities)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        if best_idx < len(thresholds):
            return thresholds[best_idx]
        return 0.05
    
    def _plot_results(self, test_pairs, auc):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(test_pairs[test_pairs['label']==1]['similarity'], bins=30, alpha=0.7, 
                     label='Citation Pairs', color='green')
        axes[0].hist(test_pairs[test_pairs['label']==0]['similarity'], bins=30, alpha=0.7, 
                     label='Random Pairs', color='red')
        axes[0].axvline(self.thresholds.get('TF-IDF', 0.05), color='blue', linestyle='--')
        axes[0].axvline(self.high_threshold, color='purple', linestyle=':')
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
        
        prec_vals, rec_vals, _ = precision_recall_curve(test_pairs['label'], test_pairs['similarity'])
        axes[2].plot(rec_vals, prec_vals, 'g-')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curve')
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def init_sbert(self, model_name="all-MiniLM-L6-v2"):
        if not SBERT_AVAILABLE:
            return False
        
        try:
            self.sbert_model = SentenceTransformer(model_name)
            all_texts = [self.text_map[pid] for pid in self.patent_ids_ordered]
            self.patent_embeddings = self.sbert_model.encode(all_texts, convert_to_tensor=True)
            self.id_to_index = {pid: i for i, pid in enumerate(self.patent_ids_ordered)}
            self.sbert_enabled = True
            return True
        except Exception as e:
            logger.error(f"SBERT failed: {e}")
            return False
    
    def compute_hybrid_similarity_batch(self, query_text, alpha=0.4, top_k=5):
        if not self.sbert_enabled:
            return self.compute_tfidf_similarity_batch(query_text, top_k)
        
        try:
            _, tfidf_raw = self.compute_tfidf_similarity_batch(query_text, top_k=len(self.patents_df))
            query_emb = self.sbert_model.encode(query_text, convert_to_tensor=True)
            sbert_raw = util.cos_sim(query_emb, self.patent_embeddings)[0].cpu().numpy()
            
            tfidf_norm = (tfidf_raw - self.tfidf_min) / (self.tfidf_max - self.tfidf_min + 1e-8)
            sbert_norm = (sbert_raw + 1) / 2
            
            hybrid_scores = alpha * tfidf_norm + (1 - alpha) * sbert_norm
            
            top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    'patent_id': self.patent_ids_ordered[idx],
                    'title': self.title_map[self.patent_ids_ordered[idx]],
                    'hybrid_sim': hybrid_scores[idx]
                })
            return results, hybrid_scores
        except Exception as e:
            return self.compute_tfidf_similarity_batch(query_text, top_k)
    
    def compute_tfidf_similarity_batch(self, query_text, top_k=5):
        clean_query = self.preprocess(query_text)
        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [{
            'patent_id': self.patent_ids_ordered[idx],
            'title': self.title_map[self.patent_ids_ordered[idx]],
            'similarity': similarities[idx]
        } for idx in top_indices]
        
        return results, similarities
    
    def predict_novelty(self, new_text, method='hybrid', alpha=0.4, top_k=5):
        if not new_text or not str(new_text).strip():
            raise ValueError("Empty input text.")
        
        if self.patents_df is None or self.text_map is None:
            raise ValueError("Dataset not loaded.")
        
        print("\n" + "="*70)
        print(f"NOVELTY PREDICTION (Method: {method.upper()})")
        print("="*70)
        print(f"\nQuery: {new_text[:200]}...")
        
        start_time = time.time()
        
        if method == 'tfidf':
            results, _ = self.compute_tfidf_similarity_batch(new_text, top_k)
            top_avg = np.mean([r['similarity'] for r in results])
        elif method == 'hybrid' and self.sbert_enabled:
            results, _ = self.compute_hybrid_similarity_batch(new_text, alpha, top_k)
            top_avg = np.mean([r['hybrid_sim'] for r in results])
        else:
            results, _ = self.compute_tfidf_similarity_batch(new_text, top_k)
            top_avg = np.mean([r['similarity'] for r in results])
        
        elapsed = time.time() - start_time
        self.inference_times[method] = elapsed
        
        threshold = self.thresholds.get('TF-IDF', 0.05)
        
        if top_avg >= self.high_threshold:
            decision = "🔴 REJECT (High Similarity - Not Novel)"
        elif top_avg >= threshold:
            decision = "🟡 INCREMENTAL (Builds on Existing Work)"
        else:
            decision = "🟢 ACCEPT (Potentially Novel)"
        
        print(f"\n📊 Results:")
        print(f"   Top-{top_k} Avg Similarity: {top_avg:.4f} ({top_avg:.1%})")
        print(f"   Inference Time: {elapsed:.3f}s")
        
        print(f"\n{'='*60}")
        print(f"TOP {top_k} SIMILAR PATENTS")
        print(f"{'='*60}")
        
        for i, r in enumerate(results, 1):
            print(f"\n{i}. Patent: {r['patent_id']}")
            print(f"   Title: {r['title'][:70]}...")
            if method == 'hybrid' and 'hybrid_sim' in r:
                print(f"   Hybrid Similarity: {r['hybrid_sim']:.4f}")
            else:
                print(f"   Similarity: {r['similarity']:.4f}")
        
        print(f"\n{'='*60}")
        print(f"🎯 DECISION: {decision}")
        print(f"{'='*60}")
        
        return decision, results
    
    def save_model(self):
        model_path = os.path.join(self.model_dir, 'tfidf_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'thresholds': self.thresholds,
                'high_threshold': self.high_threshold,
                'tfidf_min': self.tfidf_min,
                'tfidf_max': self.tfidf_max,
                'title_map': self.title_map,
                'text_map': self.text_map,
                'patent_ids_ordered': self.patent_ids_ordered
            }, f)
        print(f"✓ Model saved to {model_path}")
    
    def load_model(self):
        model_path = os.path.join(self.model_dir, 'tfidf_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.thresholds = data['thresholds']
                self.high_threshold = data['high_threshold']
                self.tfidf_min = data['tfidf_min']
                self.tfidf_max = data['tfidf_max']
                self.title_map = data.get('title_map', {})
                self.text_map = data.get('text_map', {})
                self.patent_ids_ordered = data.get('patent_ids_ordered', [])
            print(f"✓ Model loaded from {model_path}")
            return True
        return False


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*70)
    print("PATENT NOVELTY CHECK SYSTEM")
    print("NLP + Machine Learning for Prior Art Analysis")
    print("="*70)
    print(f"\nAuthor: Devika Bakshi (122CS0301)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    system = PatentNoveltySystem()
    
    if not system.load_model():
        print("No existing model found. Training new model...")
        
        try:
            patents_df, pairs_df = system.build_citation_dataset(
                patent_file="g_patent.tsv",
                abstract_file="g_patent_abstract.tsv",
                citation_file="g_us_patent_citation.tsv",
                min_citations=5, max_patents=10000, use_random_sampling=True
            )
        except Exception as e:
            print(f"Using demo data: {e}")
            patents_df, pairs_df = system._create_demo_data(num_patents=200)
        
        metrics = system.train_tfidf(pairs_df)
        if metrics is None:
            print("Training failed. Exiting.")
            return
    
    test_patent = """
    A quantum-inspired neural network architecture for real-time pattern recognition
    using adaptive resonance theory and deep reinforcement learning with attention mechanisms.
    """
    
    decision, _ = system.predict_novelty(test_patent, method='tfidf', top_k=5)
    
    print(f"\n✅ Done. Results saved to evaluation_results.png")


if __name__ == "__main__":
    main()