BERT-Contrastive-Slurm-Friendly contains Slurm-friendly .py scripts for BERT-based contrastive learning and downstream tasks.
IMPORTANT: The scripts are designed to be run on a Slurm cluster with A100/H100 GPUs. They can take very loooong if run locally.

*-featuregen.py:
    Generates combined train and test features for the downstream task.
    X_(train|test)_combined contains the randomly learned BERT CLS embeddings, tf-idf features and 17 manual linguistic features.
    X_(train|test)_combined_distance_weighted_{epoch count} contains the same features but with distance-weighted BERT CLS embeddings.

BERT-contrastive-xgb-grid.py:
    Uses grid search to find the best hyperparameters for XGBoost on the combined features.

BERT-contrastive-svm-grid.py: (deprecated)
    Uses grid search to find the best hyperparameters for SVM on the combined features.
    Defines helper functions for data visualization.
    Deprecated due to long training time and lower performance compared to SVM.

BERT-contrastive-dweighted-svm.py:
    Uses the distance-weighted features to train multiple rbf kernel SVMs with different gamma values (0.001, 'scale', 'auto').
    Trains another Logistic Regression model for comparison.


Guide:
    1. Run *-featuregen.py to generate the combined features.
    2. Run the corresponding downstream task script.


