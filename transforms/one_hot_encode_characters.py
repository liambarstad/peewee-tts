import numpy as np

class OneHotEncodeCharacters:
    def __init__(self, values: str):
        self.values = values

    def __call__(self, text):
        encoded = np.array([[]])
        for t in text:
            item = np.zeros(len(self.values))
            item[self.values.index(t)] = 1.
            if encoded.shape[1] == 0:
                encoded = np.append(encoded, item.reshape(1, -1), axis=1)
            else:
                encoded = np.append(encoded, item.reshape(1, -1), axis=0)
        return encoded
