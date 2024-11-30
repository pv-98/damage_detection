
import random
import torch
import numpy as np

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    """Convert numpy.ndarray to tensor."""

    def __call__(self, image, target):
        # Ensure image is a NumPy array before converting
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image to be np.ndarray, but got {type(image)}")
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, target

class RandomHorizontalFlip(object):
    """
    Used to flip the image horizontally in a random way and also to adjust the bounding box
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        # Ensure image is a NumPy array before flipping
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image to be np.ndarray for flipping, but got {type(image)}")
        
        if random.random() < self.prob:
            # Flip the image horizontally
            image = np.ascontiguousarray(np.flip(image, axis=1))  # Horizontal flip
            _, width, _ = image.shape

            # Adjust bounding boxes
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x coordinates
                target["boxes"] = boxes

        return image, target

def get_transform(train):
    """
    Get the list of transforms to apply to the images and targets.
    """
    transforms_list = []
    if train:
        # Apply data augmentations 
        transforms_list.append(RandomHorizontalFlip(0.5))
    #  convert to tensor
    transforms_list.append(ToTensor())
    return Compose(transforms_list)
