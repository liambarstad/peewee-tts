import os
import math
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

class Metrics:
    def __init__(self, epochs: int, per_epoch: int):
        self.epochs = epochs
        self.per_epoch = per_epoch
        self.current_step = 0
        self.data = {}
        self.aggregates = {}

    def add_step(self, values={}):
        for k in values:
            if k in self.data:
                self.data[k].append(values[k])
            else:
                self.data[k] = [ values[k] ]
        self.current_step += 1
        epoch = math.ceil(self.current_step / self.per_epoch)
        print(f'Epoch: {epoch}, Step: {self.current_step}/{self.epochs*self.per_epoch}')

    def agg_epoch(self, metric_name: str, agg_fn):
        data = self.data[metric_name]
        last_epoch_sum = agg_fn(data[-1*self.per_epoch:])
        print(f'{metric_name.upper()}: {last_epoch_sum}')
        if metric_name in self.aggregates:
            self.aggregates[metric_name].append(last_epoch_sum)
        else:
            self.aggregates[metric_name] = [ last_epoch_sum ]

    def save(self):
        for raw_metric in self.data:
            self._save_data(raw_metric+'_raw', self.data[raw_metric])
            
        for agg_metric in self.aggregates:
            aggs = self.aggregates[agg_metric]
            mlflow.log_metric(agg_metric, aggs[-1])
            self._save_graph(agg_metric, aggs)
            self._save_data(agg_metric+'_agg', aggs)

    def _save_data(self, name, data_series):
        csv_name = f'{name}.csv'
        pd.Series(data_series).to_csv(csv_name, index=False)
        mlflow.log_artifact(csv_name)
        os.remove(csv_name, dir_fd=None)

    def _save_graph(self, name, data_series):
        fig = go.Figure(data=go.Scatter(x=np.arange(len(data_series)), y=data_series))
        img_name = f'{name}.png'
        fig.write_image(img_name)
        with Image.open(img_name) as im:
            mlflow.log_image(im, img_name)
            os.remove(img_name, dir_fd=None)


