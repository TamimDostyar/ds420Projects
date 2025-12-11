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
 
def build_model_dnn(hp):
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    optimizer_choice = hp.Choice("optimizer", ["sgd", "rmsprop", "adam"])
    dropout_rate = hp.Float("dropout", 0.1, 0.5, step=0.05)
    weight_decay = hp.Choice("weight_decay", [1e-3, 1e-4])
    n_hidden = hp.Int("n_hidden", 5, 7)

    if optimizer_choice == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate)
    elif optimizer_choice == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(35,)))

    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(
            units=hp.Int("units", 64, 256, step=64),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(13, activation="softmax"))

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_model_cnn(hp):
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    optimizer_choice = hp.Choice("optimizer", ["sgd", "rmsprop", "adam"])
    weight_decay = hp.Choice("weight_decay", [1e-3, 1e-4])
    dropout_rate = hp.Float("dropout", 0.1, 0.5, step=0.1)

    n_conv_layers = hp.Int("n_conv_layers", 1, 3)
    n_dense_layers = hp.Int("n_dense_layers", 1, 3)

    if optimizer_choice == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate)
    elif optimizer_choice == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(35,)))
    model.add(tf.keras.layers.Reshape((35, 1)))

    for i in range(n_conv_layers):
        model.add(tf.keras.layers.Conv1D(
            filters=hp.Int(f"filters_{i}", 32, 128, step=32),
            kernel_size=hp.Int(f"kernel_{i}", 2, 5),
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
        ))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Flatten())

    for _ in range(n_dense_layers):
        model.add(tf.keras.layers.Dense(
            units=hp.Int("dense_units", 64, 256, step=64),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(13, activation="softmax"))

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



def final_build_dnn1():
    LEARNING_RATE = 0.0001855349762763219
    N_HIDDEN = 8
    UNITS = 256
    WEIGHT_DECAY = 0.001
    
    model = models.Sequential()
    
    model.add(layers.Input(shape=(35,)))

    for i in range(N_HIDDEN):
        model.add(layers.Dense(
            units=UNITS,
            activation="relu",
            kernel_regularizer=regularizers.L2(WEIGHT_DECAY)
        ))

        calculated_dropout = 0.05 + (0.01 * i)
        
        final_dropout = min(calculated_dropout, 0.10)
        
        model.add(layers.Dropout(final_dropout))
        


    model.add(layers.Dense(13, activation="softmax"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



def final_build_dnn2():
    LEARNING_RATE = 0.0007187981375915335
    WEIGHT_DECAY = 0.0001
    DROPOUT_RATE = 0.1
    N_CONV_LAYERS = 3
    N_DENSE_LAYERS = 3
    
    CONV_PARAMS = [
        {'filters': 32, 'kernel_size': 2},
        {'filters': 128, 'kernel_size': 5},
        {'filters': 64, 'kernel_size': 5},
    ]
    
    DENSE_UNITS = 256
    
    model = models.Sequential()
    
    model.add(layers.Input(shape=(35,)))
    
    model.add(layers.Reshape((35, 1))) 

    for i in range(N_CONV_LAYERS):
        params = CONV_PARAMS[i]
        
        model.add(layers.Conv1D(
            filters=params['filters'],
            kernel_size=params['kernel_size'],
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.L2(WEIGHT_DECAY)
        ))
        
        model.add(layers.MaxPooling1D(pool_size=2))
    
    model.add(layers.Flatten())

    for _ in range(N_DENSE_LAYERS):
        model.add(layers.Dense(
            units=DENSE_UNITS,
            activation="relu",
            kernel_regularizer=regularizers.L2(WEIGHT_DECAY)
        ))
        
        model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(13, activation="softmax"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model