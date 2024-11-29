
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data.dataloader import DamageDataset
from models.get_model import get_model
import config
from data.transforms import get_transform
import json
from tqdm import tqdm


from evaluation_metrics import get_coco_results, compute_coco_metrics, COCO

# Define a mapping from label indices to class names
CLASS_NAMES = [
    "none",
    "minor-dent",  
    "minor-scratch",  
    "moderate-broken",  
    "moderate-dent",  
    "moderate-scratch",  
    "severe-broken",  
    "severe-dent",
    "severe-scratch"   
]

def visualize_predictions(model, data_loader, device, num_images=5, threshold=0.5, save_dir=None):
    """
    Visualize model predictions on a subset of the dataset.

    Args:
        model (torch.nn.Module): Trained object detection model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        num_images (int, optional): Number of images to visualize. Defaults to 5.
        threshold (float, optional): Confidence threshold for displaying predictions. Defaults to 0.5.
        save_dir (str, optional): Directory to save the visualizations.
    """
    model.eval()
    images_so_far = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i in range(len(images)):
                if images_so_far >= num_images:
                    return  

                img = images[i].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
                img = (img * 255).astype(np.uint8)

                # Get predictions
                boxes = outputs[i]['boxes'].cpu().numpy()
                scores = outputs[i]['scores'].cpu().numpy()
                labels = outputs[i]['labels'].cpu().numpy()

                # Apply threshold
                keep = scores >= threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # Draw bounding boxes and labels on the image
                img_with_boxes = draw_boxes(img, boxes, labels, scores)

                if save_dir:
                    # Save the image
                    save_path = os.path.join(save_dir, f'prediction_{images_so_far + 1}.jpg')
                    cv2.imwrite(save_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
                    print(f"Saved visualization to {save_path}")
                else:
                    # Display the image using matplotlib
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img_with_boxes)
                    plt.axis('off')
                    plt.title(f'Image {images_so_far + 1}')
                    plt.show()

                images_so_far += 1
                if images_so_far >= num_images:
                    break  

def draw_boxes(image, boxes, labels, scores):
    """
    Draw bounding boxes and labels on an image.

    Args:
        image (np.ndarray): Image array.
        boxes (np.ndarray): Array of bounding boxes.
        labels (np.ndarray): Array of label indices.
        scores (np.ndarray): Array of confidence scores.

    Returns:
        np.ndarray: Image with drawn bounding boxes and labels.
    """
    # Create a copy of the image to draw on
    image = image.copy()

    # Define colors for different classes
    COLORS = np.random.uniform(0, 255, size=(config.NUM_CLASSES, 3))

    for idx, box in enumerate(boxes):
        color = COLORS[labels[idx] % config.NUM_CLASSES]
        label = labels[idx]
        score = scores[idx]

        x_min, y_min, x_max, y_max = box.astype(int)
        width = x_max - x_min
        height = y_max - y_min

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color.tolist(), thickness=2)

        # Prepare the label text
        if label < len(CLASS_NAMES):
            class_name = CLASS_NAMES[label]
        else:
            class_name = f"Class {label}"
        text = f"{class_name}: {score:.2f}"

        # Put the label text above the bounding box
        ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x_min, y_min - int(1.5 * text_height)), 
                      (x_min + text_width, y_min), color.tolist(), -1)
        cv2.putText(
            image, text, (x_min, y_min - int(0.5 * text_height)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1
        )

    return image

def main():
    """
    Main function to visualize predictions and compute evaluation metrics.
    """
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load the dataset
    test_dataset = DamageDataset(config.test_dir, config.test_ann, transforms=get_transform(train=False))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Initialize the model and load the trained weights
    model = get_model('faster_rcnn', config.NUM_CLASSES, pretrained=True, device=config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)

    # Visualize predictions
    visualize_predictions(model, test_loader, device, num_images=20, threshold=0.5, save_dir=os.path.join(config.OUTPUT_DIR, 'visualizations'))

    # Compute COCO Evaluation Metrics
    # Load ground truth annotations
    coco_gt = COCO(config.test_ann)

    # Generate COCO-formatted predictions
    coco_results = get_coco_results(model, test_loader, device, threshold=0.5)

    # Compute metrics
    metrics = compute_coco_metrics(coco_gt, coco_results, iou_type='bbox')

    # Print metrics
    print("\nCOCO Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # save metrics to a JSON file
    metrics_path = os.path.join(config.OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to {metrics_path}")

if __name__ == "__main__":
    main()
