import torch
from torch import nn
from torch.nn import functional as func

class SpeakerVerificationLSTMEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 projection_size,
                 embedding_size,
                 batch_size,
                 num_layers
                ):
        super(SpeakerVerificationLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.batch_size = batch_size

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

        self.W = nn.Parameter(torch.Tensor([10]))
        # contstrain to be > 0

        self.B = nn.Parameter(torch.Tensor([5]))

    def forward(self, x):
        # lstm with projection
        _, (hx, cx) = self.lstm(x)

        # linear layer w/ relu
        x = self.relu(self.linear(hx[-1]))

        # l2 normalize
        x = func.normalize(x, p=2, dim=1)

        return x


    def criterion(self, predictions, j_centroids, k_centroids):
        j_centroids = torch.tensor(j_centroids)
        k_centroids = torch.tensor(k_centroids)
        # should grad = False for j and k ?

        softmax = nn.Softmax(dim=0)
        c_j = nn.CosineSimilarity(dim=1)
        c_k = nn.CosineSimilarity(dim=2)
        cos_similarity_j = c_j(predictions, j_centroids)
        cos_similarity_k = c_k(predictions.reshape(-1, 1, predictions.shape[-1]), k_centroids)

        sji = softmax(self.W * cos_similarity_j + self.B)
        sjk = softmax(self.W * cos_similarity_k + self.B)

        loss = torch.log(torch.sum(torch.exp(sjk), dim=1)) - sji

        return loss, cos_similarity_j, cos_similarity_k



