import os
import mlflow
from torch.utils.tensorboard import SummaryWriter

class Metrics:
    def __init__(self, total_steps, model, save_model=False, out_path=None):
        self.total_steps = total_steps
        self.out_path = self._generate_artifact_path(out_path)
        self.counters = []
        self.data = {}
        self.writer = SummaryWriter(self.out_path)
        self.model = model
        self.save_model = save_model

    def add_graph(self, inputs):
        self.writer.add_graph(self.model.float(), inputs.float())

    def add_counter(self, name, func, inc):
        self.counters.append([name, func, inc]) 

    def calculate(self, curr_step, **kwargs):
        st = f'Step:{curr_step}/{self.total_steps}'
        for c in self.counters:
            name, func, inc = c[0], c[1], c[2] 
            if curr_step % inc == 0:
                result = func(**kwargs)
                self._add_data(name, result)
                self.writer.add_scalar(name, result)
                st += f', {name}:{result}'
        print(st)

    def save(self):
        for d in self.data:
            mlflow.log_metric(d, self.data[d][-1])

    def _generate_artifact_path(self, out_path=None):
        if out_path:
            return out_path
        else:
            artifact_uri = mlflow.active_run().info.artifact_uri
            cwd = os.getcwd()
            return '.'+artifact_uri.split(cwd)[-1]

    def _add_data(self, name, result):
        if name in self.data:
            self.data[name].append(result)
        else:
            self.data[name] = [result]

