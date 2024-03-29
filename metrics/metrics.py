import os
import math
from datetime import datetime
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
        print(f'{self._get_timestamp()} START RUN')

    def add_step(self, values={}, round_num=10):
        timestamp = self._get_timestamp()
        for k in values:
            if k in self.data:
                self.data[k].append(values[k])
            else:
                self.data[k] = [ values[k] ]
        self.current_step += 1
        epoch = math.ceil(self.current_step / self.per_epoch)
        step = self.current_step % self.per_epoch if self.current_step % self.per_epoch > 0 else self.per_epoch
        metrics = ', '.join([ 
            ': '.join([metric, str(round(self.data[metric][-1], round_num))]) 
            for metric in self.data 
        ])
        print(f'{timestamp} Step: {step}/{self.per_epoch}, Epoch: {epoch}/{self.epochs}, {metrics}')

    def _get_timestamp(self):
        return f'[ {str(datetime.utcnow())} ]:'

    def agg_epoch(self, metric_name: str, agg_fn):
        timestamp = self._get_timestamp()
        data = self.data[metric_name]
        last_epoch_sum = agg_fn(data[-1*self.per_epoch:])
        print(f'{timestamp} METRIC: {metric_name.upper()}: {last_epoch_sum}')
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
        print('METRICS SAVED')

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


