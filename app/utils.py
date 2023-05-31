import io
import torch
from google.cloud import storage

def load_model(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    model = torch.load(
        io.BytesIO(blob.download_as_string()),
        map_location='cpu'
    )
    if hasattr(model, 'device'):
        model.device = 'cpu'
    model.eval()
    return model
