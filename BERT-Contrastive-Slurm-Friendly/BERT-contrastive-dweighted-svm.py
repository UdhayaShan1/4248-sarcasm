# In[ ]:
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pandas as pd
import numpy as np
import torch
from sklearn.svm import SVC
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
print(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")

# In[ ]:

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

# In[ ]:
print("Loading pre-computed combined features...")
X_train_combined = load('X_train_combined_distance_weighted_100.joblib')
X_test_combined = load('X_test_combined_distance_weighted_100.joblib')

print("Final Combined Train shape:", X_train_combined.shape)
print("Final Combined Test shape:", X_test_combined.shape)


# Define function to train SVM model with grid search on gamma
def train_svm(X_train, y_train, X_test, y_test, gamma):
    print(f"Starting SVM training with gamma={gamma}...")
    start_time = time.time()

    # Create SVM model with RBF kernel
    svm_model = SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=RANDOM_STATE)

    # Fit the model
    svm_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = svm_model.predict(X_test)

    # Calculate metrics with macro averaging
    test_f1 = f1_score(y_test, y_pred, average='macro')

    # Print test metrics
    print("\nTest set metrics:")
    print(f"Macro F1 Score: {test_f1:.4f}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Log training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")

    return

# In[ ]:
train_svm(X_train_combined, y_train, X_test_combined, y_test, gamma='scale')

# In[ ]:
train_svm(X_train_combined, y_train, X_test_combined, y_test, gamma='auto')

# In[ ]:
train_svm(X_train_combined, y_train, X_test_combined, y_test, gamma=0.001)

# In[ ]:
# Sanity check: LR
def train_lr(X_train, y_train, X_test, y_test):
    print("Starting Logistic Regression training...")
    start_time = time.time()

    # Create Logistic Regression model
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000)

    # Fit the model
    lr_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = lr_model.predict(X_test)

    # Calculate metrics with macro averaging
    test_f1 = f1_score(y_test, y_pred, average='macro')

    # Print test metrics
    print("\nTest set metrics:")
    print(f"Macro F1 Score: {test_f1:.4f}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Log training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")

    return

# In[ ]:
train_lr(X_train_combined, y_train, X_test_combined, y_test)