# Breast Cancer Diagnosis ML Project

Welcome to the Breast Cancer Diagnosis ML Project! This project focuses on leveraging machine learning techniques to analyze the Breast Cancer Data Wisconsin dataset. The primary objectives include training robust machine learning models for diagnostic purposes and implementing preprocessing techniques to enhance model performance.

## Key Features

- **Data Handling:** The project efficiently handles missing values and identifies and removes outliers using the Z-score method.
- **Data Exploration:** Visualizations, such as Q-Q plots and Shapiro-Wilk tests, aid in understanding the dataset's distribution and characteristics.
- **Multicollinearity Analysis:** Explore the correlation between features and detect multicollinearity patterns.
- **Feature Selection:** Utilizes Recursive Feature Elimination (RFE) and Random Forest for optimal feature selection.
- **Model Training:** Implements Logistic Regression, Random Forest, Support Vector Classifier (SVC), and Gradient Boosting for classification tasks.
- **Hyperparameter Tuning:** Grid Search optimizes the hyperparameters of the models to enhance predictive performance.

## Getting Started

To get started, follow these steps:

1. **Data Loading:** Use `load_breast_cancer_dataset()` to load the Breast Cancer Data Wisconsin dataset.
2. **Data Preprocessing:** Apply functions like `handle_missing_values` and `handle_outliers_zscore` to clean and preprocess the data.
3. **Data Exploration:** Visualize the dataset with `visualize_qq_plot` and perform the Shapiro-Wilk test with `shapiro_wilk_test`.
4. **Feature Selection:** Utilize `feature_selection` to select relevant features for training models.
5. **Model Training:** Train and evaluate models using functions like `train_logistic_regression`, `train_random_forest`, `train_support_vector_classifier`, and `train_gradient_boosting`.

Feel free to explore and contribute to the project by running the provided functions and experimenting with different configurations.
