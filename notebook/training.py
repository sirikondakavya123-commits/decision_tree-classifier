# IMPORT LIBRARIES

import pandas as pd
import pickle
import os
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# LOAD CLEANED DATASET

df = pd.read_csv(
    "../data/cleaned_housing.csv"
)

print("Dataset Loaded Successfully")

# SELECT FEATURES AND TARGET

X = df.drop(
    "median_house_value",
    axis=1
)

# CONVERT TARGET INTO CATEGORIES
# 0: LOW, 1: MEDIUM, 2: HIGH

y = pd.cut(

    df["median_house_value"],

    bins=3,

    labels=[0, 1, 2]

)

print("Features and Target Selected")

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

# MODEL

model = RandomForestClassifier(
    random_state=42
)

# HYPERPARAMETER TUNING

params = {

    "n_estimators": [50, 100],

    "max_depth": [5, 10, 15],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2]

}

grid_search = GridSearchCV(

    estimator=model,

    param_grid=params,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=2

)

grid_search.fit(

    X_train,
    y_train

)

# BEST MODEL

best_model = grid_search.best_estimator_

print("Best Parameters")

print(
    grid_search.best_params_
)

# PREDICTION

y_pred = best_model.predict(
    X_test
)

# EVALUATION

print("Accuracy Score")

print(

    accuracy_score(
        y_test,
        y_pred
    )

)

print("Classification Report")

print(

    classification_report(
        y_test,
        y_pred
    )

)

print("Confusion Matrix")

print(

    confusion_matrix(
        y_test,
        y_pred
    )

)
# SAVE MODEL
pickle.dump(

    best_model,

    open(
        "../models/random_forest_classifier.pkl",
        "wb"
    )

)

print("Model Saved Successfully")