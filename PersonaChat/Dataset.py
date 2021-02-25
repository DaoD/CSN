import torch
from torch.utils.data import TensorDataset

class Dataset(TensorDataset):
    def __init__(self, X_utterances, X_responses, X_personas, y_labels=None):
        super(Dataset, self).__init__()
        X_utterances = torch.LongTensor(X_utterances)
        X_responses = torch.LongTensor(X_responses)
        X_personas = torch.LongTensor(X_personas)

        if y_labels is not None:
            y_labels = torch.LongTensor(y_labels)
            self.tensors = [X_utterances, X_responses, X_personas, y_labels]
        else:
            self.tensors = [X_utterances, X_responses, X_personas]

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])
