import pandas as pd
import numpy as np
import os

def preprocess(df, is_train=True, feature_columns=None):
    """
    Preprocess Titanic dataset:
    - Handle missing values
    - Feature engineering
    - Encoding
    - Ensure train-test consistency
    """

    df = df.copy()

    # Preserve PassengerId for UI & tracking
    passenger_ids = df['PassengerId'] if 'PassengerId' in df.columns else None

    # -------------------------
    # Missing Value Handling
    # -------------------------
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # -------------------------
    # Feature Engineering
    # -------------------------
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Extract title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    rare_titles = [
        'Lady','Countess','Capt','Col','Don','Dr',
        'Major','Rev','Sir','Jonkheer','Dona'
    ]
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # -------------------------
    # Drop Unused Columns
    # -------------------------
    df.drop(
        columns=['Name', 'Ticket', 'Cabin'],
        inplace=True,
        errors='ignore'
    )

    # -------------------------
    # Encoding
    # -------------------------
    df = pd.get_dummies(
        df,
        columns=['Sex', 'Embarked', 'Title', 'Pclass'],
        drop_first=True
    )

    # -------------------------
    # Ensure Feature Consistency
    # -------------------------
    if is_train:
        feature_columns = df.columns.tolist()
    else:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

    # Add PassengerId back (for UI usage)
    # Ensure PassengerId exists only once (for UI usage)
    if passenger_ids is not None and 'PassengerId' not in df.columns:
        df.insert(0, 'PassengerId', passenger_ids)


    return df, feature_columns


def save_processed_data(df, filename):
    """
    Save processed dataset to data/processed/
    """
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(f"data/processed/{filename}", index=False)
