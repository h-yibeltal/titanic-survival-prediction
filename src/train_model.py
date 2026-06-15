import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# Load processed datasets
def train_model():
    train_df = pd.read_csv("data/processed/clean_train.csv")
    test_df = pd.read_csv("data/processed/clean_test.csv")

# Separate features and target
    x_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]

    x_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(x_train,y_train)
    #log_preds = log_model.predict(x_test)
    #train_model()


    Decision_model=DecisionTreeClassifier(max_iter=1000)
    
    Decision_model.fit(x_train, y_train)
   
    log_model, decision_model=train_model()
    
