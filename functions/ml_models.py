import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

def save_model(model, model_name):
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models/")
    # Save the trained model using joblib
    joblib.dump(model, SAVE_DIR+f"{model_name}.joblib")
    print(f"{model_name} model saved successfully.")

def train_logistic_regression(x_train, y_train, x_test, y_test):
    # Hyperparameter tuning using Grid Search
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters for Logistic Regression:", grid_search.best_params_)

    # Use the best model for prediction
    best_log_reg_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_log_reg_model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return best_log_reg_model

def train_random_forest(x_train, y_train, x_test, y_test):
    # Hyperparameter tuning using Grid Search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters for Random Forest:", grid_search.best_params_)

    # Use the best model for prediction
    best_rf_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_rf_model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importances = best_rf_model.feature_importances_
    print("Feature Importances:")
    print(list(zip(x_train.columns, feature_importances)))

    # Out-of-Bag (OOB) Score
    oob_score = best_rf_model.oob_score
    print(f"Out-of-Bag (OOB) Score: {oob_score:.2f}")

    # Cross-Validation
    cv_scores = cross_val_score(best_rf_model, x_train, y_train, cv=5, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Average Cross-Validation Accuracy:", cv_scores.mean())
    return best_rf_model

def train_support_vector_classifier(x_train, y_train, x_test, y_test):
    # Hyperparameter tuning using Grid Search
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],
        'class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters for Support Vector Classifier:", grid_search.best_params_)

    # Use the best model for prediction
    best_svc_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_svc_model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Support Vector Classifier Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Cross-Validation
    cv_scores = cross_val_score(best_svc_model, x_train, y_train, cv=5, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Average Cross-Validation Accuracy:", cv_scores.mean())
    return best_svc_model

def train_gradient_boosting(x_train, y_train, x_test, y_test):
    # Hyperparameter tuning using Grid Search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0],
        'random_state': [42]
    }

    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters for Gradient Boosting:", grid_search.best_params_)

    # Use the best model for prediction
    best_gb_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_gb_model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importances = best_gb_model.feature_importances_
    print("Feature Importances:")
    print(list(zip(x_train.columns, feature_importances)))

    # Cross-Validation
    cv_scores = cross_val_score(best_gb_model, x_train, y_train, cv=5, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Average Cross-Validation Accuracy:", cv_scores.mean())
    return best_gb_model


def plot_confusion_matrix(y_true, y_pred, model_name):
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation/")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(SAVE_DIR + f'{model_name}_confusion_matrix.png')
    plt.close()

def plot_classification_report(y_true, y_pred, model_name):
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation/")
    cr = classification_report(y_true, y_pred, output_dict=True)
    df_cr = pd.DataFrame(cr).transpose()
    df_cr.to_csv(f'{model_name}_classification_report.csv')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cr.iloc[:-1, :].T, annot=True, cmap="Blues", cbar=False)
    plt.title(f'Classification Report - {model_name}')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.savefig(SAVE_DIR + f'{model_name}_classification_report.png')
    plt.close()


