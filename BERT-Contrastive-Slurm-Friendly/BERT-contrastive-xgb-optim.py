import pandas as pd
import numpy as np
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import nltk
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

# 2. Linguistic Features for TRAIN and TEST sets
from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.sparse import vstack, hstack, csr_matrix
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns

print("Loading pre-computed combined features...")
X_train_combined = load('X_train_combined.joblib')
X_test_combined = load('X_test_combined.joblib')

print("Final Combined Train shape:", X_train_combined.shape)
print("Final Combined Test shape:", X_test_combined.shape)


# --- XGBoost Model ---
xgb_max_depth = 5
xgb_lr = 0.1
xgb_estimators = 200
xgb_loss = 'logloss'
print(f"\nTraining XGBoost model on combined features... (max_depth={xgb_max_depth}, n_estimators={xgb_estimators}, learning_rate={xgb_lr}, loss={xgb_loss})")
# Create XGBoost classifier with balanced class weights via scale_pos_weight
# Determine weight based on class imbalance
pos_scale = np.sum(y_train == 0) / np.sum(y_train == 1)

xgb_model = XGBClassifier(
    scale_pos_weight=pos_scale,  # Handles class imbalance
    max_depth=xgb_max_depth,                 # Control model complexity
    learning_rate=xgb_lr,           # Learning rate
    n_estimators=xgb_estimators,            # Number of trees
    random_state=RANDOM_STATE,
    eval_metric=xgb_loss        # Evaluation metric
)

# Train the model
xgb_model.fit(X_train_combined, y_train)

print("\nEvaluating XGBoost model...")
y_pred = xgb_model.predict(X_test_combined)

print("\nClassification Report:")
# Use zero_division=0 to handle cases where a class might have no predictions in a batch/split
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=xgb_model.classes_)
print(cm)

print(f"\nMacro F1 Score: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

print("\n--- Script Finished ---")