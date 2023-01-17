import torch
from torch import nn
from torch.nn import functional as func

class SpeakerVerificationLSTMEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 projection_size,
                 embedding_size,
                 num_layers
                ):
        super(SpeakerVerificationLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            proj_size=self.projection_size,
            batch_first=True
        )

        self.linear = nn.Linear(
            in_features=self.projection_size,
            out_features=self.embedding_size
        )

        self.relu = nn.ReLU()

        self.W = 0 
        # contstrain to be > 0

        self.B = 0 

    def forward(self, x):
        # lstm with projection
        _, (hx, cx) = self.lstm(x)

        # linear layer w/ relu
        x = self.relu(self.linear(hx[-1]))

        # l2 normalize
        x = func.normalize(x, p=2, dim=1)

        return x


    def criterion(self, predictions, j_centroids, k_centroids):
        softmax = nn.Softmax()
        cos_similarity = nn.CosineSimilarity()

        import ipdb; ipdb.sset_trace()
        #Sji = softmax(model.W * cos_similarity(predictions, j_centroids) + model.B)
        #Sjk = softmax(model.W *  
        #return (-1 * Sji) + log(sum(exp(Sjk)))
        # y[:, 0] == Sji y[:, 1:] == Sjk

        #loss(eji) = 1 row in predictions
        #loss(eji) = -soft_mean(j) + log(sum(exp(k)))
        # predictions = predicted for 60 samples
        # y = centroid embedding, labels for 60 samples
        #return softmax(self.W * cos_similarity(predictions, y) + self.B)

