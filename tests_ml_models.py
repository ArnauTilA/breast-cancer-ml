def main():
    # Load and preprocess the breast cancer dataset
    df = preprocess_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.3, random_state=42)

    # Train and evaluate Logistic Regression model
    print("Training and Evaluating Logistic Regression Model:")
    train_logistic_regression(x_train, y_train, x_test, y_test)
    print("\n" + "="*50 + "\n")

    # Train and evaluate Random Forest model
    print("Training and Evaluating Random Forest Model:")
    train_random_forest(x_train, y_train, x_test, y_test)
    print("\n" + "="*50 + "\n")

    # Train and evaluate Gradient Boosting model
    print("Training and Evaluating Gradient Boosting Model:")
    train_gradient_boosting(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()