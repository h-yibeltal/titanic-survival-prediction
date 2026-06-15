import  pandas as pd 
def clean_data(df):
    df["Age"]=df["Age"].fillna(df["Age"].median())
    df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
    df.drop("Cabin",axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.info()
    return df

