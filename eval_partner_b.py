"""
Partner B: Time-Aware Holdout + 5-Fold Ordered Cross-Validation
Author: Stephanie Sookram
Dataset: Diabetes Progression
Metric: RMSE
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load dataset
data = load_diabetes(as_frame=True)
X = data.data
y = data.target

# Time-aware style holdout split
# (no shuffling – respects original ordering)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False 
)

# Model definition
model = Ridge(random_state=RANDOM_STATE)

# Fit model on training set
model.fit(X_train, y_train)

# Evaluate on test set using RMSE
y_pred = model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)

# 5-Fold ordered CV (no shuffle)
kfold = KFold(n_splits=5, shuffle=False)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kfold,
    scoring="neg_root_mean_squared_error"
)

# Convert scikit-learn's negative RMSE to positive
cv_scores = -cv_scores

# Print Results
print("Partner B Evaluation Results (Time-Aware Strategy)")
print("--------------------------------------------------")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"CV Mean RMSE: {cv_scores.mean():.4f}")
print(f"CV Std RMSE: {cv_scores.std():.4f}")
print(f"Individual Fold RMSEs: {cv_scores}")
