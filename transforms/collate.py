import torch
import numpy as np

class MaxPad:
    def __init__(self, labels_axis=None, values_axis=None):
        self.labels_axis = labels_axis
        self.values_axis = values_axis

    def __call__(self, batch):
        labels = [ b[0] for b in batch ]
        values = [ b[1] for b in batch ]
        labels = self._pad_on_axis(labels, self.labels_axis)
        values = self._pad_on_axis(values, self.values_axis)
        return torch.tensor(labels), torch.tensor(values)

    def _pad_on_axis(self, data, axis):
        if axis:
            data_shape = self._estimate_data_shape(data)
            for i in range(axis):
                # flatten to compare
                data = [ x1 for x0 in data for x1 in x0 ]
            max_length = max([ x.shape[0] for x in data ])
            for i, sample in enumerate(data):
                padding = np.zeros((max_length - sample.shape[0], *sample.shape[1:]))
                data[i] = np.append(sample, padding, axis=0)
            data = np.array(data)
            return data.reshape(*data_shape[:axis+1], *data.shape[axis:])
        else:
            return np.array(data)

    def _estimate_data_shape(self, data):
        shape = []
        subset = data.copy()
        while hasattr(subset, '__iter__'):
            shape.append(len(subset))
            subset = subset[0]
        return tuple(shape)

