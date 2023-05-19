import mlflow
import yaml

class Params:
    def __init__(self, config_path, save_model='False'):
        if save_model != 'False':
            self.save_model = True
        else:
            self.save_model = False

        with open(config_path, 'r') as file:
            self.model_config = yaml.safe_load(file)
            for section in self.model_config:
                parameters = self.model_config[section]
                setattr(self, section, parameters)

    def save(self):
        for section in self.model_config:
            parameters = self.model_config[section]
            for param in parameters:
                mlflow.log_param(f'{section}_{param}', parameters[param])
        print('PARAMETERS SAVED')