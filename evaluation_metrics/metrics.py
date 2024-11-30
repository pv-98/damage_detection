
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_coco_results(model, data_loader, device, threshold=0.5):
    """
    Generate COCO-formatted results from the model's predictions.

    Args:
        model (torch.nn.Module): Trained object detection model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.

    Returns:
        list: List of prediction dictionaries in COCO format.
    """
    model.eval()
    coco_results = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Generating Predictions"):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for img, output, target in zip(images, outputs, targets):
                image_id = int(target['image_id'].item())
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                # Apply threshold
                keep = scores >= threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min

                    coco_result = {
                        "image_id": image_id,
                        "category_id": int(label),  # Ensure category_id matches COCO format
                        "bbox": [x_min, y_min, width, height],
                        "score": float(score)
                    }
                    coco_results.append(coco_result)
    return coco_results

def compute_coco_metrics(coco_gt, coco_dt, iou_type='bbox'):
    """
    Compute COCO evaluation metrics.

    Args:
        coco_gt (COCO): COCO ground truth object.
        coco_dt (list): List of detection results in COCO format.
        iou_type (str, optional): Type of IoU to use ('bbox'). Defaults to 'bbox'.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    coco_dt = coco_gt.loadRes(coco_dt)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APs": coco_eval.stats[3],
        "APm": coco_eval.stats[4],
        "APl": coco_eval.stats[5],
        "AR1": coco_eval.stats[6],
        "AR10": coco_eval.stats[7],
        "AR100": coco_eval.stats[8],
        "ARs": coco_eval.stats[9],
        "ARm": coco_eval.stats[10],
        "ARl": coco_eval.stats[11],
    }
    return metrics
