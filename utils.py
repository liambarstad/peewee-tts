import mlflow
import yaml

class Params:
    def __init__(self, fname):
        with open(fname, 'r') as file:
            self.model_config = yaml.safe_load(file)
            for section in self.model_config:
                parameters = self.model_config[section]
                setattr(self, section, parameters)

    def save(self):
        for section in self.model_config:
            parameters = self.model_config[section]
            for param in parameters:
                mlflow.log_param(f'{section}_{param}', parameters[param])

class SpeakerCentroids:
    def __init__(self, cks={}):
        # computes the running centroids of each speaker
        self.cks = cks

    def append_data(self, speaker_id, eji):
        if speaker_id in self.cks:
            speaker_id = str(speaker_id)
            num_samples = self.cks[speaker_id][0]
            total_weight = self.cks[speaker_id][1]
            self.cks[speaker_id] = [ num_samples + 1, eji ]
        else:
            self.cks[str(speaker_id)] = [ 1, eji ]

    def get_for_speaker(self, speaker_id):
        speaker_id = str(speaker_id)
        num_samples = self.cks[speaker_id][0]
        total_weight = self.cks[speaker_id][1]
        return total_weight / num_samples

