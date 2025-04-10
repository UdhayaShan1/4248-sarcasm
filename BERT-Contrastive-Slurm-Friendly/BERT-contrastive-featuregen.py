import pandas as pd
import numpy as np
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel # Use AutoModel for hidden states
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm

df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)

df['clean_headline'] = df['headline']

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

df['pos_counts'] = df['clean_headline'].apply(get_pos_counts)
df['text_length'] = df['clean_headline'].apply(get_text_length)

df['noun_count'] = df['pos_counts'].apply(lambda x: x['noun_count'])
df['verb_count'] = df['pos_counts'].apply(lambda x: x['verb_count'])
df['adj_count'] = df['pos_counts'].apply(lambda x: x['adj_count'])
df['adv_count'] = df['pos_counts'].apply(lambda x: x['adv_count'])

import textstat
df['flesch_reading_ease'] = df['clean_headline'].apply(lambda text: textstat.flesch_reading_ease(text))
df['dale_chall_score'] = df['clean_headline'].apply(lambda text: textstat.dale_chall_readability_score(text))


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


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import random
import time
from collections import defaultdict


# In[ ]:


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# In[ ]:


df['is_sarcastic'] = df['is_sarcastic'].astype(int)
df['clean_headline'] = df['clean_headline'].astype(str)
# Reset index just in case for proper indexing later
df = df.reset_index(drop=True)


# In[ ]:


print("Splitting data into train/test sets...")
train_indices, test_indices = train_test_split(
    df.index,
    stratify=df['is_sarcastic'],
    random_state=RANDOM_STATE
)


# In[ ]:


df_train = df.loc[train_indices].reset_index(drop=True)
df_test = df.loc[test_indices].reset_index(drop=True)
y_train = df_train['is_sarcastic'].values
y_test = df_test['is_sarcastic'].values


# In[ ]:


print(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
print(f"Using device: {device}")


# In[ ]:


class TripleDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
      self.df = dataframe
      self.tokenizer = tokenizer
      self.max_length = max_length
      self.texts = dataframe['clean_headline'].values
      self.labels = dataframe['is_sarcastic'].tolist()
      self.labels_ind = {}
      for i in range(len(self.labels)):
        if self.labels[i] not in self.labels_ind:
          self.labels_ind[self.labels[i]] = []
        self.labels_ind[self.labels[i]].append(i)
    def __len__(self):
      return len(self.df)
    def __getitem__(self, idx):
      anchor_text = self.texts[idx]
      anchor_label = self.labels[idx]

      possible_positive_indices = []
      possible_negative_indices = []
      for label in self.labels_ind:
        if label != anchor_label:
          possible_negative_indices.extend(self.labels_ind[label])
        else:
          for ind in self.labels_ind[label]:
            if ind != idx:
              possible_positive_indices.append(ind)
      positive_index = -1
      if len(possible_positive_indices) > 0:
        positive_index = random.choice(possible_positive_indices)
      else:
        positive_index = idx
      negative_index = random.choice(possible_negative_indices)

      positive_text = self.texts[positive_index]
      negative_text = self.texts[negative_index]
      anchor_inputs = self.tokenizer(anchor_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
      positive_inputs = self.tokenizer(positive_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
      negative_inputs = self.tokenizer(negative_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

      return {
          'anchor': {k: v.squeeze(0) for k, v in anchor_inputs.items()},
          'positive': {k: v.squeeze(0) for k, v in positive_inputs.items()},
          'negative': {k: v.squeeze(0) for k, v in negative_inputs.items()}
      }


# In[ ]:


class BERTContrast(nn.Module):
  def __init__(self, bert_model, projection_dim=None):
    super(BERTContrast, self).__init__()
    self.bert = bert_model
    self.projection_dim = projection_dim
    self.config = self.bert.config
    self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
    if self.projection_dim:
      self.projection = nn.Sequential(
          nn.Linear(self.config.hidden_size, self.config.hidden_size),
          nn.ReLU(),
          nn.Linear(self.config.hidden_size, self.projection_dim)
      )
    else:
      self.projection = nn.Identity()

  def forward(self, input_ids, attention_mask, token_type_ids=None):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    pooled_output = outputs.last_hidden_state[:, 0, :]
    pooled_output = self.dropout(pooled_output)
    projection_output = self.projection(pooled_output)
    return projection_output



# In[ ]:


train_dataset = TripleDataset(df_train, tokenizer, 128)
test_dataset = TripleDataset(df_test, tokenizer, 128)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[ ]:


model = BERTContrast(bert_model, projection_dim=256)
model.to(device)

loss_fn = nn.TripletMarginLoss(margin=0.5)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # You can add warmup steps if needed
                                            num_training_steps=total_steps)



# In[ ]:


def get_bert_cls_embeddings(texts, model, tokenizer, device, max_length, batch_size):
  model.eval()
  cls = []
  num_batches = (len(texts) + batch_size - 1) // batch_size
  for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(
        batch_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
      outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    cls.append(cls_embeddings.cpu().numpy())
    if (i // batch_size + 1) % 50 == 0 or (i // batch_size + 1) == num_batches:
      print(f"Processed embedding batch {i // batch_size + 1}/{num_batches}")
  return np.vstack(cls)

pre_cls = get_bert_cls_embeddings(df_train['clean_headline'].tolist(), bert_model, tokenizer, device, 128, 32)
print("Pre BERT [CLS] embeddings shape:", pre_cls.shape)


# In[ ]:


def fine_tune_bert_contrastive(model, train_loader, optimizer, scheduler, loss_fn, device, epochs):
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for i, batch in enumerate(train_loader):
            anchor = {k: v.to(device) for k, v in batch['anchor'].items()}
            positive = {k: v.to(device) for k, v in batch['positive'].items()}
            negative = {k: v.to(device) for k, v in batch['negative'].items()}

            anchor_output = model(**anchor)
            positive_output = model(**positive)
            negative_output = model(**negative)

            loss = loss_fn(anchor_output, positive_output, negative_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (i + 1) % 50 == 0 or i == len(train_loader) - 1:
                print(f"  Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")


# In[ ]:


CONTRASTIVE_EPOCHS=2
fine_tune_bert_contrastive(model, train_loader, optimizer, scheduler, loss_fn, device, CONTRASTIVE_EPOCHS)


fine_tuned_base_bert_model = model.bert
fine_tuned_base_bert_model.eval()
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%eval%%%%%%%%%%%%%%%%%%%%%%%%%%%")
fine_tuned_base_bert_model.to(device)


X_bert_train = get_bert_cls_embeddings(df_train['clean_headline'].tolist(), fine_tuned_base_bert_model, tokenizer, device, 128, 32)
X_bert_test = get_bert_cls_embeddings(df_test['clean_headline'].tolist(), fine_tuned_base_bert_model, tokenizer, device, 128, 32)
print("train BERT [CLS] embeddings shape:", X_bert_train.shape)
print("test BERT [CLS] embeddings shape:", X_bert_test.shape)

# 2. Linguistic Features for TRAIN and TEST sets
from joblib import dump, load

print("\nExtracting linguistic features...")
ling_feature_names = ['text_length', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'dale_chall_score',
                      'sentiment_score', 'char_count', 'capital_char_count', 'capital_word_count',
                      'stopword_count', 'stopwords_vs_words', 'contrastive_marker', 'entropy',
                      'lexical_diversity', 'sentiment_incongruity', 'difficult_word_count']

# Ensure all ling features exist, fill missing with 0 or mean if appropriate
for col in ling_feature_names:
    if col not in df_train.columns:
        print(f"Warning: Linguistic feature '{col}' not found. Filling with 0.")
        df_train[col] = 0
        df_test[col] = 0

X_ling_train = df_train[ling_feature_names].values
X_ling_test = df_test[ling_feature_names].values
print("Linguistic TRAIN features shape:", X_ling_train.shape)
print("Linguistic TEST features shape:", X_ling_test.shape)

# 3. Combine Fine-tuned BERT [CLS] + Linguistic Features (Dense) for TRAIN and TEST
print("Combining Fine-tuned BERT [CLS] and linguistic features...")
X_gling_train = np.hstack([X_bert_train, X_ling_train])
X_gling_test = np.hstack([X_bert_test, X_ling_test])
print("Combined BERT+Ling TRAIN shape:", X_gling_train.shape)
print("Combined BERT+Ling TEST shape:", X_gling_test.shape)

# --- Feature Scaling (Fit on TRAIN, Transform TRAIN and TEST) ---
print("Scaling combined BERT+Ling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_gling_train) # Fit ONLY on training data
X_test_scaled = scaler.transform(X_gling_test)   # Transform test data

# Convert scaled features to sparse format (optional, but hstack prefers it)
X_train_sparse_scaled = csr_matrix(X_train_scaled)
X_test_sparse_scaled = csr_matrix(X_test_scaled)
print("Scaled TRAIN features shape (sparse):", X_train_sparse_scaled.shape)
print("Scaled TEST features shape (sparse):", X_test_sparse_scaled.shape)

# --- TF-IDF Features (Fit on TRAIN, Transform TRAIN and TEST) ---
print("Generating TF-IDF features...")
tfidf = TfidfVectorizer() # Optional: limit TF-IDF features (e.g., max_features=5000)

# Fit TF-IDF ONLY on the training text data
X_train_tfidf = tfidf.fit_transform(df_train['clean_headline'])
# Transform the test text data using the fitted TF-IDF vectorizer
X_test_tfidf = tfidf.transform(df_test['clean_headline'])
print("TF-IDF Train shape:", X_train_tfidf.shape)
print("TF-IDF Test shape:", X_test_tfidf.shape)

# --- Final Feature Combination ---
print("Combining TF-IDF and Scaled (Fine-tuned BERT+Ling) features...")
# Combine TF-IDF + Scaled(BERT+Ling) features using hstack
X_train_combined = hstack([X_train_tfidf, X_train_sparse_scaled])
X_test_combined = hstack([X_test_tfidf, X_test_sparse_scaled])
dump(X_train_combined, 'X_train_combined.joblib')
dump(X_test_combined, 'X_test_combined.joblib')
print("Final Combined Train shape:", X_train_combined.shape)
print("Final Combined Test shape:", X_test_combined.shape)

# # --- Logistic Regression Model ---
# print("\nTraining Logistic Regression model on combined features...")
# # Use a solver suitable for potentially large sparse data like liblinear or saga
# # Increased max_iter substantially for potentially harder convergence with high dimensions
# lr = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=RANDOM_STATE, solver='liblinear') # liblinear is often good for sparse
# lr.fit(X_train_combined, y_train) # Use y_train here
#
# print("\nEvaluating Logistic Regression model...")
# y_pred = lr.predict(X_test_combined)

# print("\nClassification Report:")
# # Use zero_division=0 to handle cases where a class might have no predictions in a batch/split
# print(classification_report(y_test, y_pred, zero_division=0)) # Use y_test here
#
# print("\nConfusion Matrix:")
# cm = confusion_matrix(y_test, y_pred, labels=lr.classes_) # Use y_test here
# print(cm)
#
# print(f"\nMacro F1 Score: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}") # Use y_test here
#
# print("\n--- Script Finished ---")




