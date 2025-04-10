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

# --- XGBoost Model ---
# xgb_max_depth = 8
# xgb_lr = 0.1
# xgb_estimators = 200
# xgb_loss = 'logloss'
# print(f"\nTraining XGBoost model on combined features... (max_depth={xgb_max_depth}, n_estimators={xgb_estimators}, learning_rate={xgb_lr}, loss={xgb_loss})")
# Create XGBoost classifier with balanced class weights via scale_pos_weight
# Determine weight based on class imbalance
pos_scale = np.sum(y_train == 0) / np.sum(y_train == 1)

# xgb_model = XGBClassifier(
#     scale_pos_weight=pos_scale,  # Handles class imbalance
#     max_depth=xgb_max_depth,                 # Control model complexity
#     learning_rate=xgb_lr,           # Learning rate
#     n_estimators=xgb_estimators,            # Number of trees
#     random_state=RANDOM_STATE,
#     eval_metric=xgb_loss        # Evaluation metric
# )

# Define parameter grid for the top 3 hyperparameters
param_grid = {
    'max_depth': [3, 5, 7, 9],           # Tree depth
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate (eta)
    'n_estimators': [100, 200, 300]      # Number of trees
}

# Base model configuration
base_model = XGBClassifier(
    scale_pos_weight=pos_scale,  # Handles class imbalance
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    use_label_encoder=False,     # Prevents warning about future deprecation
    verbosity=0                  # Reduces output verbosity
)

# Set up GridSearchCV with 5-fold cross-validation
print("\nSetting up Grid Search with 5-fold cross-validation...")
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,  # Number of cross-validation folds
    scoring='f1_macro',  # Using macro F1 score as the evaluation metric
    n_jobs=-1,  # Use all available cores
    verbose=1  # Show progress
)

# Perform grid search
print("\nPerforming grid search...")
start_time = time.time()
grid_search.fit(X_train_combined, y_train)
search_time = time.time() - start_time
print(f"Grid search completed in {search_time:.2f} seconds")

# Report best parameters and score
print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation macro F1 score: {grid_search.best_score_:.4f}")

# Evaluate on test set using best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_combined)

print("\nTest Set Evaluation:")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f"\nTest Set Macro F1 Score: {test_f1:.4f}")

# Save the best model
dump(best_model, 'best_xgboost_model.joblib')
print("\nBest model saved as 'best_xgboost_model.joblib'")

# Create a DataFrame of the grid search results for visualization
results = pd.DataFrame(grid_search.cv_results_)
dump(results, 'grid_search_results.joblib')


# Function to visualize parameter combinations
def plot_grid_search_results(results, param1, param2, param3):
    # Convert results to pivot table for visualization
    pivot_data = []

    unique_param3_values = results[f'param_{param3}'].unique()

    # For each value of param3
    for param3_value in unique_param3_values:
        # Filter results for this param3 value
        subset = results[results[f'param_{param3}'] == param3_value]

        # Create pivot table
        pivot = subset.pivot(
            index=f'param_{param1}',
            columns=f'param_{param2}',
            values='mean_test_score'
        )

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'F1 Score by {param1} and {param2} (with {param3}={param3_value})')
        plt.ylabel(param1)
        plt.xlabel(param2)
        plt.tight_layout()
        plt.savefig(f'grid_search_{param3}_{param3_value}.png')
        print(f"Saved plot for {param3}={param3_value}")


# Plot results - create one heatmap for each n_estimators value
# try:
#     plot_grid_search_results(results, 'max_depth', 'learning_rate', 'n_estimators')
#     print("Created visualization plots of grid search results")
# except Exception as e:
#     print(f"Error creating visualization: {e}")

# Print parameters ranked by importance
print("\nParameter importance (based on score variance):")
param_names = ['max_depth', 'learning_rate', 'n_estimators']
importance = {}

for param in param_names:
    # Calculate variance in scores for each parameter
    grouped = results.groupby(f'param_{param}')['mean_test_score']
    variance = grouped.mean().var()
    importance[param] = variance

# Sort by importance
for param, importance_score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{param}: {importance_score:.6f}")

print("\n--- Grid Search Completed ---")

# # Train the model
# xgb_model.fit(X_train_combined, y_train)
#
# print("\nEvaluating XGBoost model...")
# y_pred = xgb_model.predict(X_test_combined)
#
# print("\nClassification Report:")
# # Use zero_division=0 to handle cases where a class might have no predictions in a batch/split
# print(classification_report(y_test, y_pred, zero_division=0))
#
# print("\nConfusion Matrix:")
# cm = confusion_matrix(y_test, y_pred, labels=xgb_model.classes_)
# print(cm)
#
# print(f"\nMacro F1 Score: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
#
# print("\n--- Script Finished ---")



