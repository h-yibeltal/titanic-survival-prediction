import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import  StandardScaler 

def prepare_data(df):
    features=["Age","Sex","Pclass","Fare","Family_size"]
    target="Survived"
    
    x=df[features]
    y=df[target]
    x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    scaler=StandardScaler()
    num_features=["Age","Fare","Family_size"]
    x_train[num_features]=scaler.fit_transform(x_train[num_features])
    x_test[num_features]=scaler.transform(x_test[num_features])

# to save  the train and test processed   so we should do before merging the featurs and target  back togather 
    train_df=x_train.copy()
    train_df["Survived"]=y_train 
    test_df=x_test.copy()
    test_df["Survived"]=y_test
    
    train_df.to_csv("data/processed/clean_train.csv", index=False)
    test_df.to_csv("data/processed/clean_test.csv", index=False)
    print("Train and Test datasets saved.")
    return x_train, x_test, y_train, y_test
    