import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    print(" CUSTOMER CHURN PDATASET ")

    # Load THE Dataset
    df = pd.read_csv("customer_churn.csv")

    print("\nData Loaded Successfully")
    print(df.head())

    # TASK 1: DATA UNDERSTANDING

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe())

    # TASK 2: DATA CLEANING & PREPROCESSING

    print("\nData Cleaning...")

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df.dropna(inplace=True)

    # Convert Churn from Yes/No to 1/0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Data Encoding
    print("\nData Encoding...")

    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    # Churn as integer (0 or 1)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].astype(int)

    # Data Scaling
    print("\nData Scaling...")

    num_cols = df.select_dtypes(exclude='object').columns

    # Target columns remove from scaling
    if 'Churn' in num_cols:
        num_cols = num_cols.drop('Churn')

    if 'MonthlyCharges' in num_cols:
        num_cols = num_cols.drop('MonthlyCharges')

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # TASK 3: EXPLORATORY DATA ANALYSIS(EDA)
    
    print("\nEDA performing...")

    df.hist(figsize=(14, 10))
    plt.tight_layout()
    plt.show()

    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # TASK 4: SUPERVISED LEARING - REGRESSION

    if 'MonthlyCharges' in df.columns:
        print("\nRegression running...")

        X = df.drop('MonthlyCharges', axis=1)
        y = df['MonthlyCharges']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)

        y_pred = reg_model.predict(X_test)

        print("R**2 Score:", r2_score(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))

    # TASK 5: SUPERVISED LEARING - CLASSIFICATION

    if 'Churn' in df.columns:
        print("\nClassification running...")

        X = df.drop('Churn', axis=1)
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("Accuracy: {:.2f}%".format(acc * 100))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Report:\n", classification_report(y_test, y_pred))

        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance)

    # TASK 6: UNSUPERVISED LEARNING - CLUSTERING

    print("\nClustering running...")

    X = df.copy()

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    print("\nCluster Counts:")
    print(df['Cluster'].value_counts())

    plt.scatter(df['tenure'], df['MonthlyCharges'], c=df['Cluster'])
    plt.xlabel("Tenure")
    plt.ylabel("Monthly Charges")
    plt.title("Customer Segmentation using KMeans")
    plt.show()

    print("\nPROJECT SUCCESSFUL ")


if __name__ == "__main__":
    main()