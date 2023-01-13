import random
import numpy as np

class ClipShuffle:
    def __init__(self, t):
        self.t = t

    def __call__(self, data):
        # get partial utterance from each utterance with frame length t 
        partials = []

        for utterance in data:
            if utterance.shape[1] < self.t:
                # pad data if less than frame length 
                padding = np.zeros([utterance.shape[0], self.t - utterance.shape[1]])
                partial = np.append(utterance, padding, axis=1)
            else:
                # pick partial at random index
                random_ind = random.randint(0, utterance.shape[1] - self.t)
                partial = utterance[:, random_ind:random_ind+self.t]
            partials.append(partial)

        return np.array(partials)
