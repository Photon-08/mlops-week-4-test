import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.cloud import storage

# Define GCS paths
SOURCE_BUCKET_NAME = 'mlops_github_actions_test'
SOURCE_FILE_NAME = 'data/iris.csv'  # CSV must have features + target
DEST_BUCKET_NAME = 'mlops_github_actions_test'
MODEL_FILENAME = 'iris_model.joblib'

# Local file paths
LOCAL_DATA_PATH = '/tmp/iris.csv'
LOCAL_MODEL_PATH = f'/tmp/{MODEL_FILENAME}'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "banded-cumulus-466503-q7-5b181666fb91.json"


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Download a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Upload a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to bucket {bucket_name} as {destination_blob_name}.")

def train_model(data_path):
    """Train a classifier on the iris dataset."""
    df = pd.read_csv(data_path)
    
    # Assume the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained with accuracy: {acc:.2f}")

    joblib.dump(model, LOCAL_MODEL_PATH)
    print(f"Model saved to {LOCAL_MODEL_PATH}")

if __name__ == "__main__":
    # Step 1: Download data from GCS
    download_blob(SOURCE_BUCKET_NAME, SOURCE_FILE_NAME, LOCAL_DATA_PATH)

    # Step 2: Train model
    train_model(LOCAL_DATA_PATH)

    # Step 3: Upload model to GCS
    gcp_model_path = f'models/{MODEL_FILENAME}'
    upload_blob(DEST_BUCKET_NAME, LOCAL_MODEL_PATH, gcp_model_path)
    print("Training complete and model uploaded to GCS.")
    os.remove(LOCAL_DATA_PATH)  # Clean up local data file
    os.remove(LOCAL_MODEL_PATH)  # Clean up local model file
    print("Local files cleaned up.")
