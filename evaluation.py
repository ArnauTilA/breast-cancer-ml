from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from functions.ml_models import main as train_models_main

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def main():
    # Train models and get predictions
    train_models_main()

    # Replace 'your_predictions_here' with the actual predictions made by your models
    # predictions_log_reg = your_predictions_here
    # predictions_rf = your_predictions_here

    # Evaluate and plot confusion matrix for Logistic Regression
    plot_confusion_matrix(y_test, predictions_log_reg, labels=['0', '1'], title='Logistic Regression Confusion Matrix')

    # Evaluate and plot confusion matrix for Random Forest
    plot_confusion_matrix(y_test, predictions_rf, labels=['0', '1'], title='Random Forest Confusion Matrix')

if __name__ == "__main__":
    main()