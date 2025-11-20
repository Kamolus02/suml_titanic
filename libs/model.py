import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os

EMBARKED_MAP = {'C': 0, 'Q': 1, 'S': 2}


def train(data_path: Path, model_path: Path):
    df = pd.read_csv(data_path)

    if 'Cabin' in df.columns:
        df.drop(columns=['Cabin'], axis=1, inplace=True)

    df.fillna(df.mean(numeric_only=True), inplace=True)

    df.dropna(inplace=True)

    sex = pd.get_dummies(df['Sex'], drop_first=True)

    lab_enc = LabelEncoder()
    df['Embarked'] = lab_enc.fit_transform(df['Embarked'])

    df = pd.concat([df, sex], axis=1)
    df.drop(['Sex', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    forest = RandomForestClassifier(n_estimators=10, random_state=101)
    forest.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(forest, f)

    return forest.score(X, y)


def predict(passenger_data: dict, model_path: Path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    pclass = passenger_data['Pclass']
    age = passenger_data['Age']
    sibsp = passenger_data['SibSp']
    parch = passenger_data['Parch']
    fare = passenger_data['Fare']

    embarked_cleaned = EMBARKED_MAP.get(passenger_data['Embarked'], 2)

    sex_cleaned = 1 if passenger_data['Sex'].lower() == 'male' else 0

    input_df = pd.DataFrame([[
        pclass, age, sibsp, parch, fare, embarked_cleaned, sex_cleaned
    ]], columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'male'])

    prediction = model.predict(input_df)

    return int(prediction[0])