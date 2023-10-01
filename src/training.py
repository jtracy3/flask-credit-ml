import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder, 
    OneHotEncoder, 
    FunctionTransformer
)
import joblib

TRAINING_DATA = Path("data/credit.csv")
MODEL_ARTIFACTS = Path("models/artifacts/")


def train_model():

    date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    training_artifacts = MODEL_ARTIFACTS / f"{date_str}"
    Path.mkdir(training_artifacts)

    # read and split data
    data = pd.read_csv(TRAINING_DATA)
    X, y = data.loc[:, ~data.columns.isin(['bad_loan'])], data['bad_loan']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2345, stratify=y
    )

    # save model training / test
    pd.concat([X_train, y_train]).to_csv(training_artifacts / "training_data.csv")
    pd.concat([X_test, y_test]).to_csv(training_artifacts / "test_data.csv")

    # clean up categorical fields
    ordinal_features = ["account_status", "savings", "employment"]
    oh_features = [
        "credit_history",
        "purpose", 
        "personal_status_and_sex", 
        "other_debtors", 
        "property", 
        "other_installments", 
        "housing", 
        "job", 
        "telephone", 
        "foreign_worker"
    ]

    # create column preprocessors
    processor = ColumnTransformer([
        ("ordinal_transform", OrdinalEncoder(), ordinal_features),
        ("oh_transformer", OneHotEncoder(drop="first"), oh_features),
        ("credit_amount_transform", FunctionTransformer(np.log1p), ["credit_amount"])],
        remainder="passthrough"
    )

    # create pipeline
    pipeline = Pipeline([
        ("process", processor),
        ("model", XGBClassifier())
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, training_artifacts / f"pipeline.joblib")

