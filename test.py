import os
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from google.cloud import storage
import joblib
import json

from train import upload_blob  # You can import other helper functions too

with open("test_key.json", 'r') as file:
        data = json.load(file)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = data


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_model.log"),
        logging.StreamHandler()
    ]
)

# GCS Config
BUCKET_NAME = 'mlops_github_actions_test'
DATA_BLOB = 'data/iris.csv'
LOCAL_DATA_PATH = '/tmp/iris.csv'
LOCAL_MODEL_PATH = '/tmp/iris_model_test.joblib'

def download_from_gcs(bucket_name, blob_name, destination_file):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_file)
    logging.info(f"Downloaded {blob_name} from GCS bucket {bucket_name} to {destination_file}.")

def train_and_return_model(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)
    logging.info(f"Model trained and saved to {model_path}")
    
    return model, X_test, y_test

def test_model_accuracy_threshold():
    logging.info("Starting model unit test...")

    # Step 1: Download data from GCS
    download_from_gcs(BUCKET_NAME, DATA_BLOB, LOCAL_DATA_PATH)

    # Step 2: Train model and get test set
    model, X_test, y_test = train_and_return_model(LOCAL_DATA_PATH, LOCAL_MODEL_PATH)

    # Step 3: Make predictions
    y_pred = model.predict(X_test)

    # Step 4: Evaluate
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    logging.info(f"Accuracy: {acc:.2f}")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1 Score: {f1:.2f}")

    # Step 5: Assert minimum acceptable accuracy
    assert acc >= 0.70, f"Test failed: Accuracy {acc:.2f} is below 70% threshold."
    logging.info("✅ Model test passed.")
    return acc, precision, recall, f1



def generate_report(acc, precision, recall, f1):
    report = f"""
    ## 🧪 Model Evaluation Report

    - **Accuracy:** {acc:.2f}
    - **Precision:** {precision:.2f}
    - **Recall:** {recall:.2f}
    - **F1 Score:** {f1:.2f}

    """
    with open("report.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    acc, precision, recall, f1=test_model_accuracy_threshold()
    # Add this inside the test function after computing metrics
    generate_report(acc, precision, recall, f1)
    
