import gc
import torch
import mlflow

def load_model(model_uri, model_location='cpu'):
    model = mlflow.pytorch.load_model(
        model_uri=model_uri,
        map_location=torch.device(model_location)
    )
    if hasattr(model, 'device'):
        model.device = 'cpu'
    model.eval()
    return model

def create_mask_matrix(audio):
    mask_matrix = torch.rand(*audio.shape)#.to(audio.device)
    for i, speaker in enumerate(audio):
        for j, ts in enumerate(speaker):
            if ts.sum() == 0.:
                mask_matrix[i][j] = torch.ones(*ts.shape)
            else:
                mask_matrix[i][j] = torch.zeros(*ts.shape)
    return mask_matrix == 1.

def generate_stop_token_labels(mask_matrix, threshold):
    token_labels = torch.masked_fill(mask_matrix.float(), (mask_matrix > 0.0), threshold)
    end_col = torch.tensor([threshold])\
        .expand(token_labels.shape[0], 1, token_labels.shape[-1])\
        .to(token_labels.device)
    token_labels = torch.cat((token_labels, end_col), dim=1).mean(dim=2)
    return token_labels[:, 1:]

memory_dict = []
all_objs = []

def debug_memory():
    ind = len(memory_dict)
    memory_dict.append({})
    all_objs.append([])
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                memory_dict[ind][obj.data_ptr()] = obj.element_size() * obj.nelement()
            else:
                all_objs[ind].append([type(obj), obj.size()])
        except:
            pass
    import ipdb; ipdb.sset_trace()