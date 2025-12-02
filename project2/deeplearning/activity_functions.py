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