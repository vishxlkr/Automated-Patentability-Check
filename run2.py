
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data (only first time)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class PatentResearchSystem:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
        self.mid_threshold = None   # for incremental
        self.high_threshold = None  # for reject

    def preprocess(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return " ".join(words)

    def build_citation_dataset(self, patent_file, abstract_file, citation_file,
                               min_citations=5, max_patents=15000):
        """
        Build dataset where every patent has citations.
        Uses chunked reading for memory efficiency.
        """
        print("="*60)
        print("BUILDING CITATION-AWARE DATASET")
        print("="*60)

        # ------------------------------------------------------------
        # 1. Analyze citation graph in chunks to find core patents
        # ------------------------------------------------------------
        print("\n1. Analyzing citation graph (chunked)...")
        from collections import Counter
        citing_counter = Counter()
        cited_counter = Counter()

        # Determine column names from first chunk
        chunk_iter = pd.read_csv(citation_file, sep='\t', dtype=str, chunksize=500000)
        first_chunk = next(chunk_iter)
        cols = first_chunk.columns.tolist()
        print("Citation columns found:", cols)

        # Map expected column names
        patent_col = None
        cited_col = None
        for col in cols:
            if col.lower() in ['patent_id', 'citing_patent_id', 'citing_id']:
                patent_col = col
            if col.lower() in ['citation_patent_id', 'cited_patent_id', 'cited_id', 'reference_id']:
                cited_col = col

        if patent_col is None or cited_col is None:
            raise ValueError(f"Could not identify citation columns. Available: {cols}")

        # Re-initialize iterator and count
        chunk_iter = pd.read_csv(citation_file, sep='\t', dtype=str,
                                 usecols=[patent_col, cited_col], chunksize=500000)
        # Include first chunk we already read
        for chunk in [first_chunk] + list(chunk_iter):
            # Use correct column names
            citing_counter.update(chunk[patent_col])
            cited_counter.update(chunk[cited_col])

        # Get patents that are both frequently cited and frequently citing
        frequent_cited = {pid for pid, cnt in cited_counter.items() if cnt >= min_citations}
        frequent_citing = {pid for pid, cnt in citing_counter.items() if cnt >= min_citations}
        core_patents = list(frequent_cited.intersection(frequent_citing))
        print(f"Found {len(core_patents)} core patents with bidirectional citations")

        # Limit to max_patents (take top if too many)
        if len(core_patents) > max_patents:
            # Sort by citation count or just take first max_patents
            core_patents = core_patents[:max_patents]
            print(f"Limited to {max_patents} core patents for memory")

        # ------------------------------------------------------------
        # 2. Load patent data for these core patents
        # ------------------------------------------------------------
        print("\n2. Loading patent data for core patents...")
        patent_chunks = []
        for chunk in pd.read_csv(patent_file, sep='\t', dtype=str, chunksize=10000):
            mask = chunk['patent_id'].isin(core_patents)
            if mask.any():
                patent_chunks.append(chunk[mask])
        patents = pd.concat(patent_chunks, ignore_index=True)
        print(f"Loaded {len(patents)} patents from patent file")

        # Load abstracts
        abstract_chunks = []
        for chunk in pd.read_csv(abstract_file, sep='\t', dtype=str, chunksize=10000):
            mask = chunk['patent_id'].isin(patents['patent_id'])
            if mask.any():
                abstract_chunks.append(chunk[mask])
        abstracts = pd.concat(abstract_chunks, ignore_index=True)

        # Merge and keep required columns
        df = patents.merge(abstracts, on='patent_id', how='inner')
        df = df[['patent_id', 'patent_title', 'patent_abstract']].dropna()
        print(f"Final patent count after merge: {len(df)}")

        # Preprocess text immediately
        print("\nPreprocessing patent texts...")
        df['clean_text'] = (df['patent_title'] + " " + df['patent_abstract']).apply(self.preprocess)

        # ------------------------------------------------------------
        # 3. Extract citation pairs that exist within this set
        # ------------------------------------------------------------
        print("\n3. Extracting citation pairs among these patents...")
        valid_ids = set(df['patent_id'])

        # Stream citation file again, filtering pairs
        citation_pairs = []
        chunk_iter = pd.read_csv(citation_file, sep='\t', dtype=str,
                                 usecols=[patent_col, cited_col], chunksize=500000)
        for chunk in chunk_iter:
            # Rename to standard names for easier filtering
            chunk = chunk.rename(columns={patent_col: 'patent_id', cited_col: 'cited_patent_id'})
            mask = chunk['patent_id'].isin(valid_ids) & chunk['cited_patent_id'].isin(valid_ids)
            if mask.any():
                citation_pairs.append(chunk[mask])
        if citation_pairs:
            citations = pd.concat(citation_pairs, ignore_index=True)
        else:
            citations = pd.DataFrame(columns=['patent_id', 'cited_patent_id'])
        print(f"Found {len(citations)} citation pairs within the dataset")

        # ------------------------------------------------------------
        # 4. Create positive and negative pairs
        # ------------------------------------------------------------
        print("\n4. Creating training pairs...")
        if len(citations) == 0:
            raise ValueError("No citation pairs found. Try increasing max_patents or lowering min_citations.")

        # Positive pairs (citations)
        positive_pairs = citations.sample(n=min(20000, len(citations)), random_state=42)
        positive_pairs['label'] = 1

        # Negative pairs (random non-cited pairs)
        all_patents = list(valid_ids)
        negative_pairs = []
        np.random.seed(42)
        # Use a set of existing citation pairs for fast lookup
        citation_set = set(zip(citations['patent_id'], citations['cited_patent_id']))
        while len(negative_pairs) < len(positive_pairs):
            p1 = np.random.choice(all_patents)
            p2 = np.random.choice(all_patents)
            if p1 != p2 and (p1, p2) not in citation_set:
                negative_pairs.append({'patent_id': p1, 'cited_patent_id': p2})
        negative_pairs = pd.DataFrame(negative_pairs)
        negative_pairs['label'] = 0

        all_pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        print(f"Created {len(all_pairs)} pairs ({len(positive_pairs)} positive, {len(negative_pairs)} negative)")

        # Add text for both sides
        patent_texts = df.set_index('patent_id')['clean_text'].to_dict()
        all_pairs['text_1'] = all_pairs['patent_id'].map(patent_texts)
        all_pairs['text_2'] = all_pairs['cited_patent_id'].map(patent_texts)
        all_pairs.dropna(inplace=True)

        return df, all_pairs

    def train_and_evaluate(self, pairs_df):
        """Train TF-IDF model and evaluate on test set."""
        print("\n" + "="*60)
        print("TRAINING & EVALUATION")
        print("="*60)

        # Split into train/val/test (70-15-15)
        train_pairs, temp_pairs = train_test_split(
            pairs_df, test_size=0.3, random_state=42, stratify=pairs_df['label']
        )
        val_pairs, test_pairs = train_test_split(
            temp_pairs, test_size=0.5, random_state=42, stratify=temp_pairs['label']
        )
        print(f"\nTrain: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

        # Fit TF-IDF on all unique texts
        all_texts = pd.concat([pairs_df['text_1'], pairs_df['text_2']]).unique()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3),
                                          min_df=5, max_df=0.7)
        self.vectorizer.fit(all_texts)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        # Compute similarities
        def compute_sim(row):
            v1 = self.vectorizer.transform([row['text_1']])
            v2 = self.vectorizer.transform([row['text_2']])
            return cosine_similarity(v1, v2)[0][0]

        train_pairs['similarity'] = train_pairs.apply(compute_sim, axis=1)
        val_pairs['similarity'] = val_pairs.apply(compute_sim, axis=1)
        test_pairs['similarity'] = test_pairs.apply(compute_sim, axis=1)

        # ---------- Learn mid threshold from validation (max F1) ----------
        thresholds = np.arange(0.05, 0.5, 0.01)
        best_f1 = 0
        best_thresh = 0.2
        for thresh in thresholds:
            pred = (val_pairs['similarity'] >= thresh).astype(int)
            f1 = f1_score(val_pairs['label'], pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        self.mid_threshold = best_thresh
        print(f"\nOptimal mid threshold (F1): {self.mid_threshold:.3f} (F1={best_f1:.3f})")

        # ---------- Learn high threshold from positive validation pairs ----------
        pos_sims = val_pairs[val_pairs['label'] == 1]['similarity']
        if len(pos_sims) > 0:
            self.high_threshold = np.percentile(pos_sims, 75)  # 75th percentile
            print(f"High threshold (75th percentile of positives): {self.high_threshold:.3f}")
        else:
            self.high_threshold = self.mid_threshold * 1.5  # fallback
            print(f"No positive validation samples; using fallback high threshold: {self.high_threshold:.3f}")

        # Evaluate on test set using mid threshold for binary classification
        test_pred = (test_pairs['similarity'] >= self.mid_threshold).astype(int)
        acc = accuracy_score(test_pairs['label'], test_pred)
        prec = precision_score(test_pairs['label'], test_pred)
        rec = recall_score(test_pairs['label'], test_pred)
        f1 = f1_score(test_pairs['label'], test_pred)
        auc = roc_auc_score(test_pairs['label'], test_pairs['similarity'])
        cm = confusion_matrix(test_pairs['label'], test_pred)

        print("\n" + "="*60)
        print("TEST SET PERFORMANCE")
        print("="*60)
        print(f"Accuracy:  {acc:.3f} ({acc*100:.1f}%)")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        print(f"AUC-ROC:   {auc:.3f}")
        print("\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")

        # Plot results
        self.plot_results(test_pairs, test_pred, auc)

        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'auc': auc, 'mid_threshold': self.mid_threshold,
                'high_threshold': self.high_threshold, 'confusion_matrix': cm}

    def plot_results(self, test_pairs, predictions, auc):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Similarity distribution
        axes[0].hist(test_pairs[test_pairs['label']==1]['similarity'], bins=30,
                     alpha=0.7, label='Citations', color='green')
        axes[0].hist(test_pairs[test_pairs['label']==0]['similarity'], bins=30,
                     alpha=0.7, label='Non-citations', color='red')
        axes[0].axvline(self.mid_threshold, color='blue', linestyle='--',
                        label=f'Mid Threshold={self.mid_threshold:.2f}')
        if self.high_threshold:
            axes[0].axvline(self.high_threshold, color='purple', linestyle=':',
                            label=f'High Threshold={self.high_threshold:.2f}')
        axes[0].set_xlabel('Similarity')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Similarity Distribution')
        axes[0].legend()

        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(test_pairs['label'], test_pairs['similarity'])
        axes[1].plot(fpr, tpr, 'b-', label=f'AUC = {auc:.3f}')
        axes[1].plot([0,1], [0,1], 'r--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()

        # Precision-Recall curve
        from sklearn.metrics import precision_recall_curve
        prec_vals, rec_vals, _ = precision_recall_curve(test_pairs['label'],
                                                         test_pairs['similarity'])
        axes[2].plot(rec_vals, prec_vals, 'g-')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curve')
        axes[2].set_ylim([0,1])

        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150)
        plt.show()

    def predict_novelty(self, new_text, existing_df):
        """Predict novelty of a new patent text using both thresholds."""
        print("\n" + "="*60)
        print("NOVELTY PREDICTION")
        print("="*60)
        clean = self.preprocess(new_text)
        vec = self.vectorizer.transform([clean])
        sims = []
        for _, row in existing_df.iterrows():
            existing_vec = self.vectorizer.transform([row['clean_text']])
            sims.append(cosine_similarity(vec, existing_vec)[0][0])
        sims = np.array(sims)

        max_sim = np.max(sims)
        mean_sim = np.mean(sims)
        median_sim = np.median(sims)
        print(f"\nMax similarity: {max_sim:.3f}")
        print(f"Mean similarity: {mean_sim:.3f}")
        print(f"Median similarity: {median_sim:.3f}")

        if max_sim >= self.high_threshold:
            decision = "REJECT (Highly Similar)"
        elif max_sim >= self.mid_threshold:
            decision = "INCREMENTAL (Builds on Existing Work)"
        else:
            decision = "ACCEPT (Potentially Novel)"

        # Still show similar patents for transparency
        similar_idx = np.where(sims >= self.mid_threshold)[0]
        if len(similar_idx) > 0:
            top_indices = similar_idx[np.argsort(sims[similar_idx])[::-1]][:5]
            print(f"\nFound {len(similar_idx)} patents above mid threshold. Top 5:")
            for idx in top_indices:
                row = existing_df.iloc[idx]
                print(f"  {row['patent_id']} - {row['patent_title'][:80]}... (sim: {sims[idx]:.3f})")

        print(f"\nDECISION: {decision}")
        return decision, sims


def main():
    """Main execution function."""
    system = PatentResearchSystem()

    patents_df, pairs_df = system.build_citation_dataset(
        patent_file="g_patent.tsv",
        abstract_file="g_patent_abstract.tsv",
        citation_file="g_us_patent_citation.tsv",
        min_citations=5,
        max_patents=10000
    )

    results = system.train_and_evaluate(pairs_df)

    test_patent = """
    A novel quantum-inspired neural network architecture for real-time
    pattern recognition in high-dimensional data spaces. The system
    combines adaptive resonance theory with deep reinforcement learning
    to achieve unprecedented accuracy in streaming data scenarios.
    """
    decision, _ = system.predict_novelty(test_patent, patents_df)


if __name__ == "__main__":
    main()