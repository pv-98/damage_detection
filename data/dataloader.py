

import os
import torch
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
#from transforms import Compose, ToTensor, RandomHorizontalFlip  

class DamageDataset(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.transforms = transforms

        # Load annotations
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        
        self.annotations = self.coco['annotations']
        self.imgs = self.coco['images']

        
        # Map image IDs to their annotations
        self.img_id_to_annotations = self._map_img_id_to_annotations()

        # Filter out images with zero annotations
        self.valid_indices = []
        for idx, img in enumerate(self.imgs):
            img_id = img['id']
            if img_id in self.img_id_to_annotations and len(self.img_id_to_annotations[img_id]) > 0:
                self.valid_indices.append(idx)

        
        total_images = len(self.imgs)
        valid_images = len(self.valid_indices)
        excluded_images = total_images - valid_images
        print(f"Loaded {total_images} images. {valid_images} valid images will be used. {excluded_images} images excluded due to zero annotations.")

    def _map_img_id_to_annotations(self):
        img_id_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            
            if img_id not in img_id_to_annotations:
                img_id_to_annotations[img_id] = []
            img_id_to_annotations[img_id].append(ann)
        return img_id_to_annotations

    def __getitem__(self, idx):
        
        real_idx = self.valid_indices[idx]
        img_info = self.imgs[real_idx]
        img_path = os.path.join(self.root, img_info['file_name'])
        #print(f'Image path:{img_path}')
        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)  # Keep pixel values in [0, 255]

        # Get annotations for the current image
        img_id = img_info['id']
        annots = self.img_id_to_annotations.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annots:
            bbox = ann['bbox']
            # Convert [x, y, width, height] to [xmin, ymin, xmax, ymax]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            boxes.append([x_min, y_min, x_max, y_max])

            
            original_category_id = ann['category_id']
            labels.append(original_category_id)

            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])
        target['area'] = areas
        target['iscrowd'] = iscrowd

        if self.transforms:
            img, target = self.transforms(img, target)

        # Make bounding boxes to lie within image dimensions
        _, height, width = img.shape
        target['boxes'][:, 0::2].clamp_(min=0, max=width)   # x_min and x_max
        target['boxes'][:, 1::2].clamp_(min=0, max=height)  # y_min and y_max

        # Assert that there is at least one bounding box
        assert target['boxes'].shape[0] > 0, f"Image ID {img_id} has zero bounding boxes after transformations."

        return img, target

    def __len__(self):
        return len(self.valid_indices)
