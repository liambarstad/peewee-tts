import argparse
from datasets import TextAudioDataset
from utils import Params

parser = argparse.ArgumentParser(description='Trains the speaker recognition encoder, generating embeddings for different speakers')
parser.add_argument('--config-path', type=str, help='path to config .yml file')
parser.add_argument('--save-model', type=str, help='whether or not to save the model')
parser.set_defaults(save_model='False')

args = parser.parse_args().__dict__
save_model = args['save_model'] != 'False'

params = Params(args['config_path'])

if params.meta['mlflow_remote_tracking']:
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

params.save()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = TextAudioDataset()
