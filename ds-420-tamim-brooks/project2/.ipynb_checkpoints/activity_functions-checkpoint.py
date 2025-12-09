from io import StringIO
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
import kagglehub
import os
import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import math
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt

from tensorflow.keras import regularizers
def load_data():

    filename = "activity_dataset.csv"
    if not os.path.exists(filename):
        path = kagglehub.dataset_download("diegosilvadefrana/fisical-activity-dataset")
        for file in os.listdir(path):
            if file.endswith(".csv"):
                csv_path = os.path.join(path, file)
                break
        
        print(f"Loaded from Kaggle: {csv_path}")
        return pd.read_csv(csv_path)
    print(f"Loaded local file: {filename}")
    return pd.read_csv(filename)

def compute_scores(y_test, y_test_hat, verbose=False):
    accuracy = accuracy_score(y_test, y_test_hat)
    f1 = f1_score(y_test, y_test_hat, average='macro')
    recall = recall_score(y_test, y_test_hat, average='macro')
    precision = precision_score(y_test, y_test_hat, average='macro')
    
    if verbose:
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Precision: {precision:.4f}")
    
    metrics_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'F1_Score': [f1],
        'Recall': [recall],
        'Precision': [precision]
    })
    
    return metrics_df


def handle_missing_value(df, strategy):
    df_num = df.select_dtypes(include=np.number)
    df_cat = df.select_dtypes(include=object)
    
    num_model = SimpleImputer(missing_values=np.nan, strategy=strategy[0])
    arr_num = num_model.fit_transform(df_num.values)

    cat_model = SimpleImputer(missing_values=pd.NA, strategy=strategy[1])
    arr_cat = cat_model.fit_transform(df_cat.values)
    
    return (arr_num, arr_cat)


def prepare_for_train(df_train, df_test):
    
    y_column = "activityID"
    
    X_train, y_train = df_train.drop(columns=y_column), df_train[y_column]
    X_test, y_test = df_test.drop(columns=y_column), df_test[y_column]
    

    numbericalColumns = [
        "heart_rate",
        "hand temperature (°C)",
        "hand acceleration X ±16g",
        "hand acceleration Y ±16g",
        "hand acceleration Z ±16g",
        "hand gyroscope X",
        "hand gyroscope Y",
        "hand gyroscope Z",
        "hand magnetometer X",
        "hand magnetometer Y",
        "hand magnetometer Z",
        "chest temperature (°C)",
        "chest acceleration X ±16g",
        "chest acceleration Y ±16g",
        "chest acceleration Z ±16g",
        "chest gyroscope X",
        "chest gyroscope Y",
        "chest gyroscope Z",
        "chest magnetometer X",
        "chest magnetometer Y",
        "chest magnetometer Z",
        "ankle temperature (°C)",   
        "ankle gyroscope X",
        "ankle gyroscope Y",
        "ankle gyroscope Z",
        "ankle magnetometer X",
        "ankle magnetometer Y",
        "ankle magnetometer Z"
    ]
    categoricalColumns = ["PeopleId"]
    
    preprocessing_pipeline = Pipeline([
        ('imputer', ColumnTransformer([
            ("num_imputer", SimpleImputer(strategy="median"), numbericalColumns),
            ("cat_imputer", SimpleImputer(strategy="most_frequent"), categoricalColumns)
        ], remainder='drop')),
        
        ('final_transform', ColumnTransformer([
            ("num", StandardScaler(), slice(0, len(numbericalColumns))),  
            ("cat", OneHotEncoder(sparse_output=False, drop='first'), slice(len(numbericalColumns), len(numbericalColumns) + len(categoricalColumns)))
        ]))
    ])
    
    # Apply preprocessing pipeline
    X_train_final = preprocessing_pipeline.fit_transform(X_train)
    X_test_final = preprocessing_pipeline.transform(X_test)
    
    return X_train_final, y_train, X_test_final, y_test


def train_dev_split(X, y, ratio):
    n = len(X)
    ind_dev = np.asarray(random.sample(range(n), int(n*ratio)))
    ind_train = np.asarray(list(set(range(n)) - set(ind_dev)))

    return X[ind_train], y[ind_train], X[ind_dev], y[ind_dev]

def create_train_test(df, test_ratio=0.2):
    
    
    df = df.copy()

    df['stratify_col'] = df['activityID'].astype(str) + "_" + df['PeopleId'].astype(str)
    
    stratify_counts = df['stratify_col'].value_counts()
    min_count = stratify_counts.min()
    
    if min_count < 2:
        activity_counts = df['activityID'].value_counts()
        activity_min_count = activity_counts.min()
        
        if activity_min_count < 2:
            dftrain, dftest = train_test_split(df, test_size=test_ratio, random_state=42, shuffle=True)
        else:
            dftrain, dftest = train_test_split(df, test_size=test_ratio,
                                               stratify=df['activityID'],
                                               random_state=42, shuffle=True)
    else:
        dftrain, dftest = train_test_split(df, test_size=test_ratio,
                                           stratify=df['stratify_col'],
                                           random_state=42, shuffle=True)
    
    return dftrain.drop(columns=['stratify_col']), dftest.drop(columns=['stratify_col'])


# below is for DNN
def build_model(hp):
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer_choice = hp.Choice("optimizer", values=["sgd", "rmsprop", "adam"])
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.05)


    if optimizer_choice == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_choice == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(35,)), 
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.L2(0.001)),  # Layer 2
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizers.L2(0.001)),  # Layer 3
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizers.L2(0.001)),  # Layer 4
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizers.L2(0.001)),  # Layer 5
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizers.L2(0.001)),  # Layer 6
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(13, activation="softmax") 
    ])
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(35,)))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model


# 1D CNN for DNN2
def build_model_cnn(hp):
    n_hidden = hp.Int("n_hidden", min_value=2, max_value=8, default=2)
    n_conv_layers = hp.Int("n_conv_layers", min_value=1, max_value=3, default=2)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer_choice = hp.Choice("optimizer", values=["sgd", "rmsprop", "adam"])

    if optimizer_choice == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_choice == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((35, 1), input_shape=(35,))) 

    for i in range(n_conv_layers):
        model.add(tf.keras.layers.Conv1D(
            filters=hp.Int(f"filters_{i}", min_value=32, max_value=128, step=32),
            kernel_size=hp.Int(f"kernel_size_{i}", min_value=2, max_value=5),
            activation=hp.Choice(f"conv_activation_{i}", values=["relu", "tanh", "selu"]),
            padding="same"
        ))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2, padding="same"))

    model.add(tf.keras.layers.Flatten())

    for j in range(n_hidden):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f"dense_units_{j}", min_value=64, max_value=256, step=64),
            activation=hp.Choice(f"dense_activation_{j}", values=["relu", "tanh"])
        ))
        model.add(tf.keras.layers.Dropout(
            hp.Float(f"dropout_{j}", min_value=0.1, max_value=0.5, step=0.1)
        ))

    model.add(tf.keras.layers.Dense(13, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model



