import pandas as pd
import numpy as np
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import nltk

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

from tqdm import tqdm
from utils import evaluate, train_with_early_stopping

df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)

# ## Pre-Processing

# ### No Pre-Processing at all

# In[ ]:


df['clean_headline'] = df['headline']


# ### With Pre-Processing

def preprocess_text(text, action, stopword):
  #Lower Caps
  #text = text.lower()
  #Remove Punctuations
  #text = text.translate(str.maketrans('', '', string.punctuation))

  #https://www.geeksforgeeks.org/text-preprocessing-for-nlp-tasks/
    # text = text.lower()  # Lowercase
  #text = re.sub(r'\d+', '', text)  # Remove numbers
    #text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
  #text = re.sub(r'\W', ' ', text)  # Remove special characters
    # text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
  # Tokenize and remove stopwords
  words = word_tokenize(text)
  if stopword:
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

  #If stemming
  if action == "S":
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
  elif action == "L":
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
  return " ".join(words)

# Apply preprocessing to the text column
# df['clean_headline'] = df['headline'].apply(lambda text: preprocess_text(text, "", False))

# In[ ]:


def get_pos_counts(text):
    """
    Returns a dictionary with counts of certain POS tags (NOUN, VERB, ADJ, ADV)
    """
    pos_tags = pos_tag(word_tokenize(text))
    counts = {
        'noun_count': 0,
        'verb_count': 0,
        'adj_count': 0,
        'adv_count': 0
    }
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            counts['noun_count'] += 1
        elif tag.startswith('VB'):
            counts['verb_count'] += 1
        elif tag.startswith('JJ'):
            counts['adj_count'] += 1
        elif tag.startswith('RB'):
            counts['adv_count'] += 1
    return counts

def get_text_length(text):
    return len(word_tokenize(text))

import spacy
nlp = spacy.load("en_core_web_sm")

def get_ner_count(text):
    doc = nlp(text)
    return len(doc.ents)


# In[ ]:


df['pos_counts'] = df['clean_headline'].apply(get_pos_counts)
df['text_length'] = df['clean_headline'].apply(get_text_length)

df['noun_count'] = df['pos_counts'].apply(lambda x: x['noun_count'])
df['verb_count'] = df['pos_counts'].apply(lambda x: x['verb_count'])
df['adj_count'] = df['pos_counts'].apply(lambda x: x['adj_count'])
df['adv_count'] = df['pos_counts'].apply(lambda x: x['adv_count'])


# In[ ]:


import textstat
df['flesch_reading_ease'] = df['clean_headline'].apply(lambda text: textstat.flesch_reading_ease(text))
df['dale_chall_score'] = df['clean_headline'].apply(lambda text: textstat.dale_chall_readability_score(text))


# In[ ]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['clean_headline'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

def count_chars(text):
    return len(text)

def count_words(text):
    return len(text.split())

def count_capital_chars(text):
  count=0
  for i in text:
    if i.isupper():
      count+=1
  return count

def count_capital_words(text):
    return sum(map(str.isupper,text.split()))

def count_unique_words(text):
    return len(set(text.split()))

def count_exclamation(text):
    return text.count("!")

df['char_count'] = df['clean_headline'].apply(count_chars)
df['capital_char_count'] = df["clean_headline"].apply(lambda x:count_capital_chars(x))
df['capital_word_count'] = df["clean_headline"].apply(lambda x:count_capital_words(x))

df['stopword_count'] = df['clean_headline'].apply(lambda x: len([word for word in x.split() if word in stopwords.words('english')]))
df['word_count'] = df['clean_headline'].apply(count_words)
df['stopwords_vs_words'] = df['stopword_count']/df['word_count']

def has_contrastive_conjunction(text):
    contrastive_words = {"but", "although", "yet", "however", "though"}
    return int(any(word in text.split() for word in contrastive_words))

df['contrastive_marker'] = df['clean_headline'].apply(has_contrastive_conjunction)


# In[ ]:


import numpy as np
import pandas as pd
import textstat
import string
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")


# In[ ]:


from scipy.stats import entropy
def calculate_entropy(text):
    words = word_tokenize(text.lower())
    freq_dist = Counter(words)
    probs = np.array(list(freq_dist.values())) / sum(freq_dist.values())
    return entropy(probs, base=2)  # Shannon Entropy

df["entropy"] = df["clean_headline"].apply(calculate_entropy)

### 2. **Lexical Diversity (Unique Words / Total Words)**
def lexical_diversity(text):
    words = word_tokenize(text.lower())
    return len(set(words)) / len(words) if len(words) > 0 else 0

df["lexical_diversity"] = df["clean_headline"].apply(lexical_diversity)

### 6. **Wrong Words (Words Not in WordNet)**
def count_wrong_words(text):
    words = word_tokenize(text.lower())
    return sum(1 for word in words if not wordnet.synsets(word))

df["wrong_word_count"] = df["clean_headline"].apply(count_wrong_words)

### 7. **Difficult Words (Hard-to-Read Words)**
df["difficult_word_count"] = df["clean_headline"].apply(textstat.difficult_words)

### 8. **Lengthy Words (Words > 2 Characters)**
df["lengthy_word_count"] = df["clean_headline"].apply(lambda words: sum(1 for word in words if len(word) > 2))

### 9. **Two-Letter Words**
df["two_letter_words"] = df["clean_headline"].apply(lambda words: sum(1 for word in words if len(word) == 2))

### 10. **Single-Letter Words**
df["single_letter_words"] = df["clean_headline"].apply(lambda words: sum(1 for word in words if len(word) == 1))


# In[ ]:


def detect_incongruity(text):
    tokens = word_tokenize(text)
    pos_words = 0
    neg_words = 0

    for word in tokens:
        score = analyzer.polarity_scores(word)['compound']
        if score >= 0.5:
            pos_words += 1
        elif score <= -0.5:
            neg_words += 1

    # Return 1 if both positive and negative words exist → sentiment conflict
    return int(pos_words > 0 and neg_words > 0)

# Apply to the DataFrame
df['sentiment_incongruity'] = df['clean_headline'].apply(detect_incongruity)

# In[ ]:

# --- Essential Imports ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, classification_report, confusion_matrix, pairwise,
                             ConfusionMatrixDisplay) # Added ConfusionMatrixDisplay & pairwise
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix # Needed for combining features
import random
import time
from collections import defaultdict # Can be useful, though not strictly used here
import warnings

# --- Visualization Imports ---
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- Configuration ---
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 32 # Distance-Weighted can be memory intensive (I have H100!)
RANDOM_STATE = 42

# Contrastive Learning Config
CONTRASTIVE_EPOCHS = 2 # Keep low for quick testing, increase for real training (e.g., 3-5)
CONTRASTIVE_LR = 2e-5
CONTRASTIVE_MARGIN = 0.5
PROJECTION_DIM = 256
USE_PROJECTION_HEAD = True # Whether to use projection head during *contrastive* training
DISTANCE_WEIGHTING_BETA = 2.0 # Example value, might need tuning

# Visualization Config
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 350 # Slightly more iterations
VISUALIZATION_SAMPLE_SIZE = 1000

# --- Seed Everything ---
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Basic Validation
if 'is_sarcastic' not in df.columns or 'clean_headline' not in df.columns:
    raise ValueError("DataFrame must contain 'is_sarcastic' and 'clean_headline' columns.")
df['is_sarcastic'] = df['is_sarcastic'].astype(int)
df['clean_headline'] = df['clean_headline'].astype(str).fillna('')
df = df[df['clean_headline'].str.len() > 0].copy().reset_index(drop=True)
print(f"Loaded data with {len(df)} samples after cleaning.")
print(f"Class distribution:\n{df['is_sarcastic'].value_counts(normalize=True)}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Split ---
print("\nSplitting data into train/test sets...")
train_indices, test_indices = train_test_split(
    df.index,
    stratify=df['is_sarcastic'],
    random_state=RANDOM_STATE
)
df_train = df.loc[train_indices].reset_index(drop=True)
df_test = df.loc[test_indices].reset_index(drop=True)
y_train = df_train['is_sarcastic'].values
y_test = df_test['is_sarcastic'].values
print(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")

# --- Load Tokenizer and Base BERT Model ---
print(f"\nLoading tokenizer and base BERT model: {BERT_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model_base = AutoModel.from_pretrained(BERT_MODEL_NAME)
bert_model_base.to(device)

# --- Embedding Generation Function ---
def get_bert_cls_embeddings(texts, model, tokenizer, device, max_length, batch_size, desc="Generating embeddings"):
    all_cls_embeddings = []
    model.eval()
    num_samples = len(texts)
    print(f"  {desc} for {num_samples} texts in batches of {batch_size}...")
    num_batches = (num_samples + batch_size - 1) // batch_size
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_texts = [t if isinstance(t, str) and len(t) > 0 else "[PAD]" for t in batch_texts]
            if not batch_texts: continue
            inputs = tokenizer(batch_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_cls_embeddings.append(cls_embeddings)
            if (i // batch_size + 1) % 50 == 0 or (i // batch_size + 1) == num_batches:
                 progress = (i // batch_size + 1) / num_batches; elapsed = time.time() - start_time
                 eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
                 print(f"    Processed batch {i // batch_size + 1}/{num_batches} ({progress*100:.1f}%) | ETA: {eta:.1f}s")
    print(f"  Finished generating embeddings in {time.time() - start_time:.2f}s.")
    if not all_cls_embeddings:
        hidden_size = model.config.hidden_size if hasattr(model, 'config') else 768
        return np.array([]).reshape(0, hidden_size)
    return np.vstack(all_cls_embeddings)

# --- Generate Embeddings BEFORE Fine-tuning (for Train and Test) ---
print("\nGenerating ORIGINAL BERT [CLS] embeddings (BEFORE fine-tuning)...")
inference_batch_size = BATCH_SIZE * 2 # Can use larger batch for inference


# --- Contrastive Learning Setup ---

# --- Dataset Class ---
class SarcasmDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe, self.tokenizer, self.max_length = dataframe, tokenizer, max_length
        self.texts, self.labels = dataframe['clean_headline'].tolist(), dataframe['is_sarcastic'].values
    def __len__(self): return len(self.dataframe)
    def __getitem__(self, index):
        text, label = self.texts[index], self.labels[index]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()} # Remove batch dim
        return inputs, torch.tensor(label, dtype=torch.long)

# --- Contrastive Model Wrapper ---
class ContrastiveBERT(nn.Module):
    def __init__(self, bert_model, projection_dim=None, use_projection=True): # Added use_projection flag
        super().__init__()
        self.bert = bert_model
        self.use_projection = use_projection # Store the flag
        self.projection_dim = projection_dim
        self.config = bert_model.config
        if self.use_projection and self.projection_dim:
            print(f"Using projection head: {self.config.hidden_size} -> {self.projection_dim}")
            self.projection_head = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.ReLU(), nn.Linear(self.config.hidden_size, self.projection_dim))
        else:
            print("Using direct BERT CLS embeddings (no projection head during contrastive training).")
            self.projection_head = nn.Identity()
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None and 'token_type_ids' in self.bert.forward.__code__.co_varnames: bert_inputs['token_type_ids'] = token_type_ids
        outputs = self.bert(**bert_inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.projection_head(cls_embedding) # Apply projection head (or identity)

# --- Loss Function: Distance-Weighted Sampling ---
def distance_weighted_triplet_loss(labels, embeddings, margin, beta, squared=False, epsilon=1e-8):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    if squared: pairwise_dist = pairwise_dist.pow(2)
    batch_size = labels.size(0); anchor_indices = torch.arange(batch_size, device=labels.device)
    valid_triplets_mask = torch.zeros(batch_size, dtype=torch.bool, device=labels.device)
    final_pos_dist = torch.zeros(batch_size, device=labels.device)
    final_neg_dist = torch.zeros(batch_size, device=labels.device)
    num_valid_anchors = 0
    for i in range(batch_size):
        anchor_label = labels[i]
        pos_mask = (labels == anchor_label) & (anchor_indices != i)
        if not torch.any(pos_mask): continue
        pos_distances = pairwise_dist[i][pos_mask]
        hardest_positive_dist = torch.max(pos_distances) # Farthest positive
        neg_mask = labels != anchor_label
        if not torch.any(neg_mask): continue
        neg_indices = anchor_indices[neg_mask]
        neg_distances = pairwise_dist[i][neg_mask]
        weights = torch.exp(-beta * torch.clamp(neg_distances, min=epsilon))
        if torch.sum(weights) < epsilon:
            print(f"Warning: Anchor {i} near-zero weights, using uniform neg sampling.", end='\r') # Less verbose warning
            sampled_neg_local_idx = torch.randint(0, len(neg_indices), (1,), device=labels.device).item()
        else:
            sampled_neg_local_idx = torch.multinomial(weights, 1).item()
        sampled_neg_dist = neg_distances[sampled_neg_local_idx]
        valid_triplets_mask[i] = True
        final_pos_dist[i] = hardest_positive_dist
        final_neg_dist[i] = sampled_neg_dist
        num_valid_anchors += 1
    if num_valid_anchors == 0: return torch.tensor(0.0, device=embeddings.device, requires_grad=True), 0.0
    valid_mask = valid_triplets_mask
    triplet_loss_values = final_pos_dist[valid_mask] - final_neg_dist[valid_mask] + margin
    triplet_loss = torch.mean(torch.relu(triplet_loss_values))
    fraction_valid_triplets = num_valid_anchors / batch_size
    return triplet_loss, fraction_valid_triplets

# --- Fine-tuning Function ---
def fine_tune_bert_contrastive(model, dataloader, optimizer, scheduler, device, epochs, margin, distance_beta):
    model.train(); start_time = time.time()
    print("\n--- Starting Contrastive Fine-tuning (Distance-Weighted Sampling) ---")
    for epoch in range(epochs):
        epoch_loss, epoch_frac_active, num_batches_processed = 0.0, 0.0, 0
        epoch_start_time = time.time(); print(f"\nEpoch {epoch+1}/{epochs}")
        for i, batch_data in enumerate(dataloader):
            optimizer.zero_grad(); loss, frac_active = None, 0.0
            inputs, labels = batch_data
            if len(labels) <= 1: print(f"Warning: Skipping batch {i+1} size {len(labels)} <= 1."); continue
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            embeddings = model(**inputs)
            loss, frac_active = distance_weighted_triplet_loss(labels, embeddings, margin, beta=distance_beta)
            if loss is not None and not torch.isnan(loss) and loss.item() > 1e-8: # Check loss > 0 with tolerance
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step(); scheduler.step()
                epoch_loss += loss.item(); epoch_frac_active += frac_active
                num_batches_processed += 1
            elif loss is not None and torch.isnan(loss): print(f"Warning: NaN loss batch {i+1}.")
            if (i + 1) % 25 == 0 or i == len(dataloader) - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Batch {i+1}/{len(dataloader)} | Loss: {loss.item() if loss is not None else 0.0:.4f} | LR: {current_lr:.2e} | Frac Active: {frac_active:.3f}")
        avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else 0
        avg_frac_active = epoch_frac_active / num_batches_processed if num_batches_processed > 0 else 0
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f} | Avg Frac Active: {avg_frac_active:.3f} | Time: {time.time() - epoch_start_time:.2f}s")
    print(f"--- Contrastive Fine-tuning Finished ({time.time() - start_time:.2f}s) ---")

# --- Prepare for Contrastive Fine-tuning ---
print("\nPreparing for contrastive fine-tuning using Distance-Weighted Sampling...")
train_dataset = SarcasmDataset(df_train, tokenizer, MAX_LENGTH)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True if device == torch.device("cuda") else False, drop_last=True)
bert_model_to_finetune = AutoModel.from_pretrained(BERT_MODEL_NAME)
contrastive_model = ContrastiveBERT(bert_model_to_finetune, projection_dim=PROJECTION_DIM, use_projection=USE_PROJECTION_HEAD).to(device)
optimizer = optim.AdamW(contrastive_model.parameters(), lr=CONTRASTIVE_LR, eps=1e-8)
total_steps = len(train_dataloader) * CONTRASTIVE_EPOCHS
num_warmup_steps = int(0.1 * len(train_dataloader)) # Warmup over 10% of first epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

# --- Run Contrastive Fine-tuning ---
fine_tune_bert_contrastive(contrastive_model, train_dataloader, optimizer, scheduler, device, CONTRASTIVE_EPOCHS, CONTRASTIVE_MARGIN, DISTANCE_WEIGHTING_BETA)

# Extract the fine-tuned base BERT model (always needed for downstream tasks)
fine_tuned_base_bert_model = contrastive_model.bert
fine_tuned_base_bert_model.eval()
fine_tuned_base_bert_model.to(device)

# --- Generate Embeddings AFTER Fine-tuning ---
print("\nGenerating Fine-tuned BERT [CLS] embeddings (AFTER fine-tuning)...")
X_bert_train_after_ft = get_bert_cls_embeddings(df_train['clean_headline'].tolist(), fine_tuned_base_bert_model, tokenizer, device, MAX_LENGTH, inference_batch_size, desc="Generating 'After FT' Train Embeddings")
print("Fine-tuned BERT [CLS] TRAIN embeddings shape:", X_bert_train_after_ft.shape)
X_bert_test_after_ft = get_bert_cls_embeddings(df_test['clean_headline'].tolist(), fine_tuned_base_bert_model, tokenizer, device, MAX_LENGTH, inference_batch_size, desc="Generating 'After FT' Test Embeddings")
print("Fine-tuned BERT [CLS] TEST embeddings shape:", X_bert_test_after_ft.shape)

# --- Downstream Task: Feature Preparation ---

ling_feature_names = ['text_length', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'dale_chall_score',
                      'sentiment_score', 'char_count', 'capital_char_count', 'capital_word_count',
                      'stopword_count', 'stopwords_vs_words', 'contrastive_marker', 'entropy',
                      'lexical_diversity', 'sentiment_incongruity', 'difficult_word_count']

# 1. Linguistic Features
print("\nExtracting linguistic features...")
X_ling_train = df_train[ling_feature_names].values
X_ling_test = df_test[ling_feature_names].values
print(f"Linguistic shapes: Train {X_ling_train.shape}, Test {X_ling_test.shape}")

# 2. TF-IDF Features
print("\nGenerating TF-IDF features...")
tfidf = TfidfVectorizer() # Consistent max_features
X_train_tfidf = tfidf.fit_transform(df_train['clean_headline'])
X_test_tfidf = tfidf.transform(df_test['clean_headline'])
print(f"TF-IDF shapes: Train {X_train_tfidf.shape}, Test {X_test_tfidf.shape}")
tfidf_feature_names_ = tfidf.get_feature_names_out() # Store TF-IDF feature names
num_tfidf_features_ = len(tfidf_feature_names_)

# --- Prepare Features for AFTER Fine-tuning Model ---
print("\nPreparing features for AFTER Fine-tuning model...")
# Combine Fine-tuned BERT [CLS] + Linguistic Features
X_gling_train_after = np.hstack([X_bert_train_after_ft, X_ling_train])
X_gling_test_after = np.hstack([X_bert_test_after_ft, X_ling_test])
# Scale Combined BERT+Ling Features (AFTER) - Use a SEPARATE scaler
scaler_after = StandardScaler()
X_gling_train_scaled_after = scaler_after.fit_transform(X_gling_train_after)
X_gling_test_scaled_after = scaler_after.transform(X_gling_test_after)
X_train_sparse_scaled_gling_after = csr_matrix(X_gling_train_scaled_after)
X_test_sparse_scaled_gling_after = csr_matrix(X_gling_test_scaled_after)
# Final Combination (AFTER)
X_train_combined_after = hstack([X_train_tfidf, X_train_sparse_scaled_gling_after])
X_test_combined_after = hstack([X_test_tfidf, X_test_sparse_scaled_gling_after])
print("Combined features (AFTER FT) shapes: Train {}, Test {}".format(X_train_combined_after.shape, X_test_combined_after.shape))

# --- Export Combined Features ---
from joblib import dump, load
dump(X_train_combined_after, "X_train_combined_distance_weighted.joblib")
dump(X_test_combined_after, "X_test_combined_distance_weighted.joblib")