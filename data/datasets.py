import os
import nibabel as nib
import numpy as np
import torch
from torch.utils import data

from torch.utils.data import TensorDataset, DataLoader, Dataset

class BrainTumour(Dataset):
    def __init__(self, split='Train', data_dir = './',
        fold=0, transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        folds = [0,1,2,3]
        folds.remove(fold)

        # Get list of images
        images = sorted(os.listdir(images_path:= data_dir+'imagesTr/'))
        labels = sorted(os.listdir(labels_path:= data_dir+'labelsTr/'))
        assert(len(images) == len(labels))
        
        # partition dataset
        images = np.array_split(images, len(folds)+1)
        labels = np.array_split(labels, len(folds)+1)

        if split == 'Valid':
            data0 = list(map(lambda img: nib.load(images_path+img), images[folds[0]]))
            target0 = list(map(lambda label: nib.load(labels_path+label), labels[folds[0]]))
            self.data, self.targets = data0, target0

        elif split == 'Train':
            data0 = np.concatenate((images[folds[1]], images[folds[2]]))
            target0 = np.concatenate((labels[folds[1]], labels[folds[2]]))

            self.data = list(map(lambda img: nib.load(images_path+img), data0))
            self.targets = list(map(lambda label: nib.load(labels_path+label), target0))
        else:
            data0 = list(map(lambda img: nib.load(images_path+img), images[fold]))
            target0 = list(map(lambda label: nib.load(labels_path+label), labels[fold]))
            self.data, self.targets = data0, target0
        
            

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        loaded_img = self.data[index].get_fdata()
        loaded_label = self.targets[index].get_fdata()

        # Transforms labels to C-boolean
        image, label = loaded_img, (loaded_label > 0).astype(int)
        
        if self.transform is not None:
            sample = {'image': image, 'mask': label}
            transformed = self.transform(sample)
            image = transformed["image"]
            label = transformed["mask"]
        
        label = label.type(torch.FloatTensor)
        self.data[index].uncache()
        self.targets[index].uncache()
        return image, label.squeeze()

            
            
                