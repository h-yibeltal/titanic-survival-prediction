import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(df: pd.DataFrame):
    df_copy = df.copy()  # work on a copy to avoid modifying original

    # 1️⃣ Survival by Gender
    plt.figure(figsize=(8,5))
    sns.countplot(x="Sex", hue="Survived", data=df_copy)
    plt.title("Survival by Gender")
    plt.tight_layout()
    plt.savefig("outputs/figures/survival_vs_gender.png")
    plt.show()
    plt.clf()

    # 2️⃣ Survival by Passenger Class
    plt.figure(figsize=(8,5))
    sns.countplot(x="Pclass", hue="Survived", data=df_copy)
    plt.title("Survival by Passenger Class")
    plt.tight_layout()
    plt.savefig("outputs/figures/survival_vs_class.png")
    plt.show()
    plt.clf()

    # 3️⃣ Age Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df_copy["Age"], bins=30, kde=True)
    plt.title("Age Distribution of Passengers")
    plt.tight_layout()
    plt.savefig("outputs/figures/age_distribution.png")
    plt.show()
    plt.clf()

    # 4️⃣ Feature Correlation Heatmap
    df_copy["Sex"] = df_copy["Sex"].map({"male":0, "female":1})
    plt.figure(figsize=(10,6))
    numeric_df = df_copy.select_dtypes(include=["number"])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/figures/feature_correlation_heatmap.png")
    plt.show()
    plt.clf()