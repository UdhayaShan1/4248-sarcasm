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
X_train_combined = load('X_train_combined.joblib')
X_test_combined = load('X_test_combined.joblib')

print("Final Combined Train shape:", X_train_combined.shape)
print("Final Combined Test shape:", X_test_combined.shape)


# Define function to train SVM model with grid search on gamma
def train_svm_with_grid_search(X_train, y_train, X_test, y_test):
    print("Starting SVM training with grid search on gamma...")
    start_time = time.time()

    # Define parameter grid for gamma
    param_grid = {
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100]
    }

    # Create SVM model with RBF kernel
    svm_model = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)

    # Set up GridSearchCV with macro F1 score
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        scoring='f1_macro',  # Changed from 'f1' to 'f1_macro'
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # Fit the grid search model
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_svm = grid_search.best_estimator_

    # Print the best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best Macro F1 score on validation data: {grid_search.best_score_:.4f}")  # Updated label

    # Evaluate on test set
    y_pred = best_svm.predict(X_test)

    # Calculate metrics with macro averaging
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')  # Specify macro averaging
    test_precision = precision_score(y_test, y_pred, average='macro')  # Specify macro averaging
    test_recall = recall_score(y_test, y_pred, average='macro')  # Specify macro averaging

    # Print test metrics
    print("\nTest set metrics:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1 Score: {test_f1:.4f}")  # Updated label
    print(f"Macro Precision: {test_precision:.4f}")  # Updated label
    print(f"Macro Recall: {test_recall:.4f}")  # Updated label

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

    # Create results dict for exporting
    results = {
        'model': best_svm,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'grid_search_results': grid_search.cv_results_,
        'test_predictions': y_pred,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'confusion_matrix': cm,
        'training_time': training_time
    }

    return results

# In[ ]:
# Define visualization helper functions
def create_visualization_data(results):
    """
    Create and save visualization data for local viewing
    """
    # Extract grid search results
    gamma_values = results['grid_search_results']['param_gamma'].data
    f1_macro_scores = results['grid_search_results']['mean_test_score']

    # Create DataFrame for visualization
    viz_data = pd.DataFrame({
        'gamma': gamma_values,
        'macro_f1_score': f1_macro_scores  # Updated column name to reflect macro F1
    })

    # Add confusion matrix
    viz_data_cm = pd.DataFrame(
        results['confusion_matrix'],
        columns=['Predicted Negative', 'Predicted Positive'],
        index=['Actual Negative', 'Actual Positive']
    )

    # Add metrics with updated names to reflect macro averaging
    metrics = {
        'accuracy': results['test_accuracy'],
        'macro_f1_score': results['test_f1'],  # Updated key name
        'macro_precision': results['test_precision'],  # Updated key name
        'macro_recall': results['test_recall'],  # Updated key name
        'best_gamma': results['best_params']['gamma'],
        'training_time': results['training_time']
    }

    # Create visualization dictionary
    visualization_data = {
        'gamma_performance': viz_data,
        'confusion_matrix': viz_data_cm,
        'metrics': metrics
    }

    # Save visualization data
    dump(visualization_data, 'svm_macro_f1_visualization_data.joblib')  # Updated filename
    print("Visualization data saved to 'svm_macro_f1_visualization_data.joblib'")

    return visualization_data


# Functions to be used locally for visualization
def plot_gamma_performance(visualization_data):
    """
    Plot the performance of different gamma values
    """
    gamma_perf = visualization_data['gamma_performance']

    plt.figure(figsize=(12, 6))

    # Convert string gamma values to numeric for proper plotting
    gamma_values = []
    for g in gamma_perf['gamma']:
        if g == 'scale':
            gamma_values.append(-1)
        elif g == 'auto':
            gamma_values.append(-2)
        else:
            gamma_values.append(float(g))

    # Create new dataframe with numeric gamma values
    plot_df = pd.DataFrame({
        'gamma': gamma_values,
        'macro_f1_score': gamma_perf['macro_f1_score']  # Updated column name
    })

    # Sort by gamma value for better visualization
    plot_df = plot_df.sort_values('gamma')

    # Replace -1 and -2 with 'scale' and 'auto' for display
    x_labels = []
    for g in plot_df['gamma']:
        if g == -1:
            x_labels.append('scale')
        elif g == -2:
            x_labels.append('auto')
        else:
            x_labels.append(str(g))

    # Plot all points
    plt.plot(range(len(plot_df)), plot_df['macro_f1_score'], marker='o', linestyle='-', color='blue')

    # Find index and value of highest macro F1 score
    best_idx = plot_df['macro_f1_score'].idxmax()
    best_value = plot_df.loc[best_idx, 'macro_f1_score']
    best_gamma = plot_df.loc[best_idx, 'gamma']
    best_pos = plot_df.index.get_loc(best_idx)

    # Highlight the best point in red
    plt.plot(best_pos, best_value, marker='o', markersize=10, color='red')

    # Add label with the actual value
    plt.annotate(f'{best_value:.4f}',
                 xy=(best_pos, best_value),
                 xytext=(best_pos, best_value + 0.02),  # Offset label slightly above
                 ha='center',
                 fontweight='bold')

    plt.xticks(range(len(plot_df)), x_labels, rotation=45)
    plt.title('Macro F1 Score by Gamma Value (SVM with RBF Kernel)')
    plt.xlabel('Gamma Value')
    plt.ylabel('Macro F1 Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('svm_gamma_macro_f1_performance.png')
    plt.close()


def plot_confusion_matrix(visualization_data):
    """
    Plot the confusion matrix
    """
    cm = visualization_data['confusion_matrix'].values

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix - SVM with RBF Kernel')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png')
    plt.close()


def plot_metrics_summary(visualization_data):
    """
    Create a bar plot of performance metrics
    """
    metrics = visualization_data['metrics']

    # Extract the metrics we want to plot with updated names
    metric_names = ['accuracy', 'macro_f1_score', 'macro_precision', 'macro_recall']
    display_names = ['Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall']
    metric_values = [metrics[name] for name in metric_names]

    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = plt.bar(display_names, metric_values, color=colors)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.ylim(0, 1.1)
    plt.title('SVM Performance Metrics (Macro Averaging)')
    plt.ylabel('Score')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('svm_macro_metrics_summary.png')
    plt.close()


def create_visualization_report():
    """
    Function to be run locally to generate all visualizations
    """
    # Load the visualization data
    visualization_data = load('svm_macro_f1_visualization_data.joblib')

    # Generate all plots
    plot_gamma_performance(visualization_data)
    plot_confusion_matrix(visualization_data)
    plot_metrics_summary(visualization_data)

    print("All visualizations have been generated!")

# In[ ]:

# Main execution flow
print("Starting SVM with RBF kernel training...")

# Train SVM model with grid search
svm_results = train_svm_with_grid_search(X_train_combined, y_train, X_test_combined, y_test)

# Save the model
print("Saving the SVM model...")
dump(svm_results['model'], 'svm_model_rbf.joblib')
print("Model saved as 'svm_model_rbf.joblib'")

# Create and save visualization data
visualization_data = create_visualization_data(svm_results)

# Save all results
dump(svm_results, 'svm_results.joblib')
print("All results saved to 'svm_results.joblib'")
print("\n--- Script Finished ---")

# In[ ]:
create_visualization_report()

# In[ ]:
best_svm_model = load('svm_model_rbf.joblib')
y_pred = best_svm_model.predict(X_test_combined)
print(f1_score(y_test, y_pred, average='macro'))