import torch
from torch import nn

#   pca of each centroid
#   euclidian distance between centroid and pred
#   avg euclidian distance between other centroids and pred

def contrast_metric(predictions, cos_similarity_j, cos_similarity_k, **kwargs):
    with torch.no_grad():
        return (1 - torch.sigmoid(cos_similarity_j).sum()) + torch.max(torch.sigmoid(cos_similarity_k), dim=1).values.sum()

def loss_metric(loss, **kwargs):
    return loss.sum() / len(loss) 
