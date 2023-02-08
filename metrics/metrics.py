import os
import mlflow
import plotly.express as px
import pandas as pd
from PIL import Image

class Metrics:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.counters = []
        self.data = {}

    def add_counter(self, name, func, inc):
        self.counters.append([name, func, inc]) 

    def calculate(self, epoch, curr_step, **kwargs):
        st = f'Epoch: {epoch}, Step:{curr_step}/{self.total_steps}'
        for c in self.counters:
            name, func, inc = c[0], c[1], c[2] 
            if curr_step == 1 or curr_step % inc == 0:
                result = func(**kwargs)
                self._add_data(name, result)
                st += f', {name}:{result}'
        print(st)

    def save(self):
        for d in self.data:
            data = self.data[d]
            mlflow.log_metric(d, data[-1])
            inc = next(filter(lambda c: c[0] == d, self.counters))[-1]
            data_df = pd.DataFrame({ 'step': [ i * inc for i, _ in enumerate(data) ], d: data})
            self.save_graph(d, data_df)
            self.save_data(d, data_df)

    def save_graph(self, name, data_df):
        fig = px.line(data_df, x='step', y=name, text=name, title=name)
        fig.update_traces(textposition='top left')
        img_name = f'{name}.png'
        fig.write_image(img_name)
        with Image.open(img_name) as im:
            mlflow.log_image(im, img_name)
            os.remove(img_name, dir_fd=None)

    def save_data(self, name, data_df):
        csv_name = f'{name}.csv'
        data_df.to_csv(csv_name, index=False)
        mlflow.log_artifact(csv_name)
        os.remove(csv_name, dir_fd=None)

    def _add_data(self, name, result):
        if name in self.data:
            self.data[name].append(result)
        else:
            self.data[name] = [result]

