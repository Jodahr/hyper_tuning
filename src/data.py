from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pandas as pd


def getData(dataPath, label):
    data = joblib.load(dataPath)
    y = data[label]
    X = data.drop(label, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    return {'train': (X_train, y_train), 'test': (X_test, y_test)}
