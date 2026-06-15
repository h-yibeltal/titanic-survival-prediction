import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

def preprocess_titanic(df:pd.DataFrame)-> pd.DataFrame:
    
   #1,Family features
    df['Family_size'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = (df['Family_size'] == 1).astype(int)
    
    # 2 Encode Sex
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    
    # 3 One-hot encode Embarked
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)
    # 4
    # 5 Drop Name column
    df.drop('Name', axis=1, inplace=True)
    
    # 6 Scale continuous features
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
     
    return df