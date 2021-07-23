import torch
import numpy as np
import random

class ZeroPad(object):
    """Pads the item to a specified shape

    Args:
        output_size (int): Desired output shape
    """

    def __init__(self, output_size, seed=None):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else: 
            assert len(output_size) == 3
            self.output_size = output_size
        if seed:
            random.seed(seed)

    def __call__(self, sample) -> dict:

        image, mask = sample['image'], sample['mask']
        pad_shape = ()
        for axis in range(len(mask.shape)):

            axis_size = mask.shape[axis]
            desired_size = self.output_size[axis]
            start, rest = divmod(desired_size-axis_size, 2)
            start, end = random.sample((start, start+rest), 2)

            pad_shape += ((start, end),)
        
        # asume channel is tail
        padded_img = np.pad(image, pad_shape+((0,0),))
        padded_mask = np.pad(mask, pad_shape)

        return {'image': padded_img, 'mask': padded_mask}

class Norm(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample) -> dict:
        image, mask = sample['image'], sample['mask']

        norm_image = image
        for i in range(image.shape[-1]):
            norm_image[...,i] = (image[...,i] - image[...,i].min())/(image[...,i].max()-image[...,i].min())
        return {'image': norm_image, 'mask': mask}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x D x C
        # torch image: C x H x W x D
        image = np.moveaxis(image, -1, 0)
        return {'image': torch.from_numpy(image),
        'mask': torch.from_numpy(mask)}