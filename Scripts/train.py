import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


def prepare_data():
    df = pd.read_csv("../Data/clean_data.csv")
    
    # Not used in model
    company_name = df["company_name"]
    df = df.drop("company_name", axis=1)

    df.fillna(0, inplace=True)

    y = df["listing_gains"]   # True = Loss, False = Gain
    del df["listing_gains"]

    X = df
    return X, y 


def train_model(X, y):
    # Best hyperparameters you provided (cleaned up for XGBoost)
    best_params = {
    "C": 0.017787658410143285,
    "max_iter": 200,
    "penalty": "l1",
    "solver": "saga"
}


    model = LogisticRegression(**best_params)
    model.fit(X, y)

    return model


def save_model(model):
    with open("model.bin", "wb") as f_out:
        pickle.dump(model, f_out)
    print("========= Model saved to model.bin =========")


if __name__ == "__main__":
    X, y = prepare_data()
    model = train_model(X, y)
    save_model(model)
