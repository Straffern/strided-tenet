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

        # prepend path
        images = np.vectorize(lambda x: np.core.defchararray.add(images_path, x))(images)
        labels = np.vectorize(lambda x: np.core.defchararray.add(labels_path, x))(labels)

        if split == 'Valid':
            self.data, self.targets = images[folds[0]], labels[folds[0]]

        elif split == 'Train':
            self.data = np.concatenate((images[folds[1]], images[folds[2]]))
            self.targets = np.concatenate((labels[folds[1]], labels[folds[2]]))

        else:
            self.data, self.targets = images[fold], labels[fold]
            

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        # load proxy files
        proxy_img = nib.load(self.data[index])
        proxy_label = nib.load(self.targets[index])
        
        # read files into memory
        loaded_img = np.asarray(proxy_img.dataobj)
        loaded_label = np.asarray(proxy_label.dataobj)

        # Transforms labels to C-boolean
        image, label = loaded_img, (loaded_label > 0).astype(int)
        
        if self.transform is not None:
            sample = {'image': image, 'mask': label}
            transformed = self.transform(sample)
            image = transformed["image"]
            label = transformed["mask"]
        
        # label = label.type(torch.IntTensor)
        # image = image.type(torch.IntTensor)
        label = label.type(torch.FloatTensor)


        return image, label.squeeze()

            
            
                