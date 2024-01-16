# Import necessary libraries
from functions.preprocesing import load_breast_cancer_dataset, handle_missing_values, handle_outliers_zscore, feature_selection

def preprocess_data():
    SAVE_DIR = "C:/Users/PORTATIL/PycharmProjects/BreastCancer/evaluation/"
    # Load breast cancer dataset
    breast_cancer_data = load_breast_cancer_dataset()

    # Handle missing values
    df_cleaned = handle_missing_values(breast_cancer_data)

    # Handle outliers using Z-score
    df_no_outliers_zscore = handle_outliers_zscore(df_cleaned)

    # Check multicollinearity among features
    #check_multicollinearity(df_no_outliers_zscore.drop('target', axis=1))

    # Feature selection and standardization
    x_standardized = feature_selection(df_no_outliers_zscore.drop('target', axis=1), df_no_outliers_zscore['target'])

    # Display the final cleaned and standardized DataFrame
    print("Final Cleaned and Standardized DataFrame:")
    print(x_standardized.head())

    x_standardized.to_csv(SAVE_DIR + 'preprocessed_data.csv', index=False)
    # Return the final DataFrame
    return x_standardized
