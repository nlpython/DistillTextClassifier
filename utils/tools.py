import torch
from torch import nn
import random
import os
import numpy as np
from torch.utils.data import Dataset

class InputFeature(object):
    """
    BertClassification's input feature
    """
    def __init__(self, input_ids, mask_ids, type_ids, label, text):
        self.input_ids = input_ids
        self.mask_ids = mask_ids
        self.type_ids = type_ids
        self.label = label
        self.text = text

    def __str__(self):
        return f'input_ids: {self.input_ids}, mask_ids: {self.mask_ids}, type_ids: {self.type_ids}, label: {self.label}'


class LstmFeatures(object):
    """
    LstmClassifier's input feature
    """
    def __init__(self, input_ids, label, text):
        self.input_ids = input_ids
        self.label = label
        self.text = text

    def __str__(self):
        return f'input_ids: {self.input_ids}, labels: {self.label}'


class ClassifierDataset(Dataset):
    """
    Teacher model dataset
    """

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    @staticmethod
    def collate_fn(batch):
        input_ids = [feature.input_ids for feature in batch]
        mask_ids = [feature.mask_ids for feature in batch]
        type_ids = [feature.type_ids for feature in batch]
        label = [feature.label for feature in batch]
        text = [feature.text for feature in batch]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(type_ids, dtype=torch.long),
        }, torch.tensor(label, dtype=torch.long), text

class LstmDataset(Dataset):
    """
    Student model dataset
    """

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    @staticmethod
    def collate_fn(batch):
        input_ids = [feature.input_ids for feature in batch]
        label = [feature.label for feature in batch]
        text = [feature.text for feature in batch]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
        }, torch.tensor(label, dtype=torch.long), text

def seed_everything(seed=1234):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def save_model(model, logger, path):
    torch.save(model.state_dict(), path)
    logger.info(f'Saved model to {path}')




