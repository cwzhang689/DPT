import torch
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class SIMSData(Dataset):
    def __init__(self, data_path, split, drop_rate, full_data=False):
        super(SIMSData, self).__init__()
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        self.data = data[split]
        self.split = split
        self.drop_rate = drop_rate
        self.full_data = full_data
        self.orig_dims = [
            self.data['text'][0].shape[1],
            self.data['audio'][0].shape[1],
            self.data['vision'][0].shape[1]
        ]

    def get_dim(self):
        return self.orig_dims
    
    def get_seq_len(self):
        return self.data['text'][0].shape[0], self.data['audio'][0].shape[0], self.data['vision'][0].shape[0]

    def __len__(self):
        return self.data['audio'].shape[0]
    
    def get_missing_mode(self):
        if self.full_data:
            return 6
        if random.random() < self.drop_rate:
            return random.randint(0, 5) 
        else:
            return 6


    def __getitem__(self, idx):
        L_feat = torch.tensor(self.data['text'][idx]).float()
        A_feat = torch.tensor(self.data['audio'][idx]).float()
        V_feat = torch.tensor(self.data['vision'][idx]).float()
        label = torch.tensor(self.data['regression_labels'][idx]).float()
        X = (L_feat, A_feat, V_feat)
        missing_code = self.get_missing_mode()

        return X, label, missing_code
    
    from torch.nn.utils.rnn import pad_sequence

class MOSIData(Dataset):
    def __init__(self, dataset_path, split_type="train", drop_rate=0.6, full_data=False):
        super(MOSIData, self).__init__()
        dataset = pickle.load(open(dataset_path, "rb"))
        self.vision = torch.tensor(dataset[split_type]["vision"].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]["text"].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]["audio"].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]["labels"].astype(np.float32)).cpu().detach()
        self.drop_rate = drop_rate
        self.full_data = full_data
        self.n_modalities = 3  # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        return self.labels.shape[1], self.labels.shape[2]

    def get_missing_mode(self):
        if self.full_data:
            return 6
        if random.random() < self.drop_rate:
            return random.randint(0, 5)
        else:
            return 6

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        return X, Y, self.get_missing_mode()


    def collate_fn(self, batch):
        max_length = 350
        A = [
            torch.cat(
                [
                    sample[0][1],
                    torch.zeros(
                        (max_length - len(sample[0][1]), sample[0][1].shape[1]),
                        device="cpu",
                    ),
                ]
            )
            for sample in batch
        ]
        V = [sample[0][2] for sample in batch]
        L = [sample[0][0] for sample in batch]
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)

        label = torch.tensor([sample[1] for sample in batch]).float()
        missing_code = torch.tensor([sample[2] for sample in batch]).float()

        # 确保转换为 FloatTensor
        L, A, V = L.float(), A.float(), V.float()

        X = (L, A, V)

        return X, label, missing_code