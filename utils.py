import mlflow
import yaml

class Params:
    def __init__(self, fname):
        with open(fname, 'r') as file:
            model_config = yaml.safe_load(file)
            for section in model_config:
                parameters = model_config[section]
                setattr(self, section, parameters)
                for param in parameters:
                    mlflow.log_param(f'{section}_{param}', parameters[param])

