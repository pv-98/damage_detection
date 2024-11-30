import unittest
from unittest.mock import patch, mock_open
import json
import torch
import numpy as np
import os
from io import BytesIO
from PIL import Image
import cv2
from dataloader import DamageDataset

# Define a simple transform for testing
def dummy_transform(img, target):
    # Convert image to tensor (C, H, W)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img, target

class TestDamageDataset(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "images": [
            {"id": 1, "file_name": "image1.jpg"},
            {"id": 2, "file_name": "image2.jpg"},
            {"id": 3, "file_name": "image3.jpg"}
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "bbox": [10, 20, 30, 40], "category_id": 1, "area": 1200, "iscrowd": 0},
            {"id": 102, "image_id": 1, "bbox": [15, 25, 35, 45], "category_id": 2, "area": 1575, "iscrowd": 0},
            {"id": 103, "image_id": 2, "bbox": [50, 60, 70, 80], "category_id": 3, "area": 5600, "iscrowd": 0}
        ]
    }))
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_initialization_and_length(self, mock_cvtColor, mock_imread, mock_file):
        """
        Test that the dataset initializes correctly and excludes images without annotations.
        """
        # Mock image reading to return a valid image array
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        # Mock color conversion to return the same image
        mock_cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        dataset = DamageDataset(root='/fake_root', ann_file='fake_ann.json', transforms=dummy_transform)

        # Images 1 and 2 have annotations, image 3 does not
        self.assertEqual(len(dataset), 2, "Dataset should contain 2 valid images.")

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "images": [
            {"id": 1, "file_name": "image1.jpg"}
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "bbox": [10, 20, 30, 40], "category_id": 1, "area": 1200, "iscrowd": 0}
        ]
    }))
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_getitem(self, mock_cvtColor, mock_imread, mock_file):
        """
        Test that __getitem__ returns the correct image tensor and target dictionary.
        """
        # Create a dummy image array
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = img_array
        mock_cvtColor.return_value = img_array

        dataset = DamageDataset(root='/fake_root', ann_file='fake_ann.json', transforms=dummy_transform)

        img, target = dataset[0]

        # Check that the image is a tensor of shape (3, 100, 100)
        self.assertIsInstance(img, torch.Tensor, "Image should be a torch.Tensor.")
        self.assertEqual(img.shape, (3, 100, 100), "Image tensor should have shape (3, 100, 100).")

        # Check target dictionary keys
        expected_keys = {'boxes', 'labels', 'image_id', 'area', 'iscrowd'}
        self.assertTrue(expected_keys.issubset(target.keys()), "Target should contain the correct keys.")

        # Check contents of target
        self.assertEqual(target['boxes'].shape, (1, 4), "Boxes tensor should have shape (1, 4).")
        self.assertEqual(target['labels'].shape, (1,), "Labels tensor should have shape (1,).")
        self.assertEqual(target['image_id'].shape, (1,), "Image ID tensor should have shape (1,).")
        self.assertEqual(target['area'].shape, (1,), "Area tensor should have shape (1,).")
        self.assertEqual(target['iscrowd'].shape, (1,), "iscrowd tensor should have shape (1,).")

        # Verify bounding box coordinates
        expected_box = torch.tensor([10, 20, 40, 60], dtype=torch.float32)
        self.assertTrue(torch.equal(target['boxes'][0], expected_box), "Bounding box coordinates are incorrect.")

    

# Run the tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)