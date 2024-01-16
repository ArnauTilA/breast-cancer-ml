from functions.ml_models import train_logistic_regression,train_random_forest,train_gradient_boosting,save_model,plot_classification_report,plot_confusion_matrix
from tests.tests_preprocesing import preprocess_data
from sklearn.model_selection import train_test_split

def train_evaluate_models(x_train, x_test, y_train, y_test):

    # Train and evaluate Logistic Regression model
    print("Training and Evaluating Logistic Regression Model:")
    best_log_reg_model = train_logistic_regression(x_train, y_train, x_test, y_test)
    save_model(best_log_reg_model, "logistic_regression")

    y_pred_log_reg = best_log_reg_model.predict(x_test)
    plot_confusion_matrix(y_test, y_pred_log_reg, "logistic_regression")
    plot_classification_report(y_test, y_pred_log_reg, "logistic_regression")

    print("\n" + "=" * 50 + "\n")

    # Train and evaluate Random Forest model
    print("Training and Evaluating Random Forest Model:")
    best_rf_model = train_random_forest(x_train, y_train, x_test, y_test)
    save_model(best_rf_model, "random_forest")

    y_pred_rf = best_rf_model.predict(x_test)
    plot_confusion_matrix(y_test, y_pred_rf, "random_forest")
    plot_classification_report(y_test, y_pred_rf, "random_forest")

    print("\n" + "=" * 50 + "\n")

    # Train and evaluate Gradient Boosting model
    print("Training and Evaluating Gradient Boosting Model:")
    best_gb_model = train_gradient_boosting(x_train, y_train, x_test, y_test)
    save_model(best_gb_model, "gradient_boosting")

    y_pred_gb = best_gb_model.predict(x_test)
    plot_confusion_matrix(y_test, y_pred_gb, "gradient_boosting")
    plot_classification_report(y_test, y_pred_gb, "gradient_boosting")


df = preprocess_data()


x_train, x_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.3,
                                                    random_state=42)

train_evaluate_models(x_train, x_test, y_train, y_test)