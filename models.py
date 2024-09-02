import os
from data_split import perform_train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Random Forest Classifier
def train_random_forest_classifier(X_train, y_train):
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    return model

# Random Forest Regressor
def train_random_forest_regression(X_train, y_train):
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    return model

# Decision Tree Classifier
def train_decision_tree_classifier(X_train, y_train):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    return model

# Decision Tree Regressor
def train_decision_tree_regression(X_train, y_train):
    model = DecisionTreeRegressor(random_state=123)
    model.fit(X_train, y_train)
    return model

# Linear Regression
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Logistic Regression Classifier
def train_logistic_regression_classifier(X_train, y_train):
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    return model

def evaluate_model_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)  # Probabilities for ROC/AUC

    results = {}
    results["Accuracy"] = accuracy_score(y_test, y_pred)
    results["Precision"] = precision_score(y_test, y_pred, average='weighted')
    results["Recall"] = recall_score(y_test, y_pred, average='weighted')
    results["F1 Score"] = f1_score(y_test, y_pred, average='weighted')
    results["Confusion Matrix"] = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for better JSON compatibility
    
    # ROC and AUC Calculation
    if len(set(y_test)) == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        results["AUC"] = roc_auc
        
    else:  # Multiclass classification
        y_test_bin = label_binarize(y_test, classes=range(len(set(y_test))))
        n_classes = y_test_bin.shape[1]

        roc_auc_dict = {}
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc_dict[i] = auc(fpr, tpr)

        results["AUC per class"] = roc_auc_dict

    return results

def print_performance_metrics_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    results = {}
    results["Mean Squared Error (MSE)"] = mean_squared_error(y_test, y_pred)
    results["Root Mean Squared Error (RMSE)"] = np.sqrt(results["Mean Squared Error (MSE)"])
    results["Mean Absolute Error (MAE)"] = mean_absolute_error(y_test, y_pred)
    results["R-squared (R2)"] = r2_score(y_test, y_pred)
    
    return results

# Display available models
def display_models(models):
    print("Available Models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
