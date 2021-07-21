import torch


class MoADataset:
    def __init__(self, features, targets):
        self.features = features.values
        self.targets = targets.values
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        idx_features = torch.tensor(self.features[idx], dtype=torch.float)
        idx_targets = torch.tensor(self.targets[idx], dtype=torch.float)
        return idx_features, idx_targets


class TestMoADataset:
    def __init__(self, features):
        self.features = features.values
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        idx_features = torch.tensor(self.features[idx], dtype=torch.float)
        return idx_features