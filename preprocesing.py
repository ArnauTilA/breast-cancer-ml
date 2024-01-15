import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer
import seaborn as sns
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter


def load_breast_cancer_dataset():
    return load_breast_cancer()


def handle_missing_values(data):
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['target'] = data.target

    nan_count_per_column = df.isna().sum()
    #print("NaN count per column:")
    #print(nan_count_per_column)

    df_cleaned = df.dropna()
    #print("Rows that contained Nan cleaned")

    return df_cleaned


def handle_outliers_zscore(df_cleaned):
    z_scores = np.abs((df_cleaned - df_cleaned.mean()) / df_cleaned.std())
    df_no_outliers_zscore = df_cleaned[(z_scores < 3).all(axis=1)]

    return df_no_outliers_zscore


def visualize_qq_plot(df):
    plt.figure(figsize=(10, 6))
    for feature in df.columns:
        sm.qqplot(df[feature], line='s', label=feature, alpha=0.5)

    plt.title("Q-Q Plot for All Variables")
    plt.legend()
    plt.show()


def shapiro_wilk_test(df):
    for feature in df.columns:
        stat, p_value = shapiro(df[feature])
        if p_value > 0.05:
            print(f"Shapiro-Wilk test for {feature}: stat={stat:.4f}, p-value={p_value:.4f}")
            print(f"The {feature} column appears to be normally distributed.\n")
        else:
            pass


def check_multicollinearity(df):
    features_mean = df.columns[1:10]
    features_se = df.columns[9:19]
    features_worst = df.columns[19:]

    corr_mean = df[features_mean].corr()
    visualize_heatmap(corr_mean, features_mean)

    corr_se = df[features_se].corr()
    visualize_heatmap(corr_se, features_se)

    corr_worst = df[features_worst].corr()
    visualize_heatmap(corr_worst, features_worst)


def visualize_heatmap(corr_matrix, feature_names):
    g = sns.heatmap(corr_matrix, cbar=True, annot=True, annot_kws={'size': 15}, fmt='.2f', square=True, cmap='coolwarm')
    g.set_xticklabels(rotation=90, labels=feature_names, size=15)
    g.set_yticklabels(rotation=0, labels=feature_names, size=15)
    g.set_xticks(np.arange(.5, 10.5, 1))
    plt.rcParams["figure.figsize"] = (17, 17)
    plt.show()


def feature_selection(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    #number_selected = plot_rfecv(x_train, y_train)
    number_selected = 11

    most_appearing_features = find_most_appearing_features(x_train, y_train)
    print('Most appearing features :')
    print(Counter(most_appearing_features).most_common(number_selected))

    selected_features = [feature for feature, _ in Counter(most_appearing_features).most_common(11)]
    x_selected = x[selected_features]

    x_standardized = standardize_features(x_selected)
    x_standardized['target'] = list(y)

    return x_standardized


def plot_rfecv(x_train, y_train):
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")

    #number_of_random_states = 100
    number_of_random_states = 1
    average_optimal = np.zeros(30)

    for rs in tqdm(range(number_of_random_states)):
        clf_rf_4 = RandomForestClassifier(random_state=rs)
        rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5, scoring='recall')
        rfecv = rfecv.fit(x_train, y_train)
        average_optimal += np.asarray(rfecv.cv_results_['mean_test_score'])
    average_optimal /= number_of_random_states
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), average_optimal)
    print("Number of features selected :", np.argmax(average_optimal) + 1)
    print("Evaluation of the optimal recall :", np.max(average_optimal))
    plt.show()
    return np.argmax(average_optimal) + 1

def find_most_appearing_features(x_train, y_train):
    most_appearing_features = []

    for rs in range(10):
        clf_rf_2 = RandomForestClassifier(random_state=rs)
        rfe = RFE(estimator=clf_rf_2, n_features_to_select=11, step=1)
        rfe = rfe.fit(x_train, y_train)
        most_appearing_features.append(x_train.columns[rfe.support_].tolist())

    most_appearing_features = [item for sublist in most_appearing_features for item in sublist]

    return most_appearing_features


def standardize_features(x_selected):
    scaler = StandardScaler()
    x_standardized = scaler.fit_transform(x_selected)
    x_standardized = pd.DataFrame(x_standardized, columns=x_selected.columns)

    return x_standardized


# Usage
"""
breast_cancer_data = load_breast_cancer_dataset()
df_cleaned = handle_missing_values(breast_cancer_data)
df_no_outliers_zscore = handle_outliers_zscore(df_cleaned)
visualize_qq_plot(df_no_outliers_zscore)
shapiro_wilk_test(df_no_outliers_zscore)
check_multicollinearity(df_no_outliers_zscore)
x_standardized = feature_selection(df_no_outliers_zscore.drop('target', axis=1), df_no_outliers_zscore['target'])
"""