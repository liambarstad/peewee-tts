import io
import os
import torch
from google.cloud import storage

def load_model(bucket_name, blob_name):
    #path = os.path.join(os.getcwd(), 'peewee-tts-d5d337076562.json')
    #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path

    storage_client = storage.Client(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
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
