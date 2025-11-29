"""
Partner A: Random Holdout + 5-Fold Standard CV
Author: Mark Young
Dataset: Diabetes Progression
Metric: RMSE

This script performs the following steps:
1. Loads the Diabetes dataset from scikit-learn.
2. Splits the data into training and test sets (80/20) using a fixed random seed.
3. Trains a Ridge regression model on the training data.
4. Evaluates the model on the test set using RMSE.
5. Performs 5-fold standard cross-validation on the training set.
6. Converts CV scores to RMSE and prints all results.
"""


import numpy as np
from math import sqrt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes



# Set random seed for reproducibility

# Ensures that splits and model training are consistent across runs
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


#Load the dataset from scikit-learn
# as_frame=True returns a pandas DataFrame for easier handling
data = load_diabetes(as_frame=True)

# X: features (10 numeric columns)
# y: target (disease progression)
X, y = load_diabetes(as_frame=True, return_X_y=True)

# Split data into training and test sets, 80% training, 20% test
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Initialize and train Ridge regression model.Ridge regression is used with default alpha (no tuning)
# random_state ensures reproducibility for internal computations
model = Ridge(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Evaluate model on test set
# Predict values for test set
y_pred = model.predict(X_test)

# Compute Mean Squared Error (MSE). Common way to measure how wrong a regression model's predictions are. 
mse = mean_squared_error(y_test, y_pred)

# Convert MSE to Root Mean Squared Error (RMSE). This is better because it gives the units back to the original target variable.

test_rmse = sqrt(mse)

# Perform 5-Fold Standard Cross-Validation on training set. This gives a better estimate of model performance.
# shuffle=True ensures folds are randomized
# random_state ensures reproducibility
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# cross_val_score computes the score for each fold
# Use neg_mean_squared_error because scikit-learn expects "higher is better"
cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=kfold,
    scoring="neg_mean_squared_error"
)

# cross_val_score returns negative MSE → convert to positive RMSE
cv_scores = np.sqrt(np.abs(cv_scores))


# Print all results
# 
print(f"Test RMSE: {test_rmse:.4f}")         # Single test set RMSE
print(f"CV Mean RMSE: {cv_scores.mean():.4f}")  # Average RMSE across folds
print(f"CV Std RMSE: {cv_scores.std():.4f}")    # Variability of RMSE across folds
print("Individual Folds:", cv_scores)           # RMSE for each fold
