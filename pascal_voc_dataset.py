import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

class PascalVOCDataset(Dataset):
    def __init__(self, images_path:str, labels_path:str, transforms, preprocess):
        # LABELS
        df = pd.read_csv(labels_path, index_col=0)
        keys = df.index.tolist()
        data_tensor = torch.tensor(df.iloc[:, :].values.flatten(), dtype=torch.float32)
        self.image_path_to_label_dict = {key: data_tensor[i:i+len(df.columns)] for i, key in enumerate(keys)}
        self.num_to_label_dict = {i: key for i, key in enumerate(df.columns)}
        self.num_labels = len(self.num_to_label_dict)

        # IMAGES
        self.images_path = [images_path + path for path in sorted(os.listdir(images_path)) if path in self.image_path_to_label_dict]

        # PREPROCESS
        self.transforms = transforms
        self.preprocess = preprocess
    
    def __getitem__(self, idx):
        path = self.images_path[idx]
        image_id = path.split('/')[-1]
        label = self.image_path_to_label_dict[image_id]

        image = Image.open(path)
        image = ToTensor()(image)
        if self.transforms:
            image = self.transforms(image)
        image = self.preprocess(image)

        return image, label

    def __len__(self):
        return len(self.images_path)
