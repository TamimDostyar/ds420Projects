from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
import kagglehub
import os

    
def load_data():
    filename = "activity_dataset.csv"

    # If file not found locally, download from Kaggle
    if not os.path.exists(filename):
        path = kagglehub.dataset_download("diegosilvadefrana/fisical-activity-dataset")
        
        # Find the first CSV file in the downloaded folder
        for file in os.listdir(path):
            if file.endswith(".csv"):
                csv_path = os.path.join(path, file)
                break
        
        print(f"Loaded from Kaggle: {csv_path}")
        return pd.read_csv(csv_path)
    
    # Otherwise, load from local file
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