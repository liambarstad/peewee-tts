import torch
import numpy as np

class MaxPad:
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, batch):
        outputs = []
        for i, _ in enumerate(batch[0]):
            if not self.axis or i in self.axis:
                col = [ b[i] for b in batch ]
                maximum_val = max([ c.shape[0] for c in col ])
                samples = []
                for c in col:
                    padding = np.zeros((maximum_val - c.shape[0], *c.shape[1:]))
                    padded_samp = np.append(c, padding, axis=0)
                    samples.append(padded_samp)
                outputs.append(torch.tensor(np.array(samples)))
            else:
                outputs.append(torch.tensor([ b[i] for b in batch ]))
        return outputs

