
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from training_pipeline.engine import train_one_epoch, compute_validation_loss, generate_validation_visualizations
from training_pipeline.utils import collate_fn
from models.get_model import get_model
from data.dataloader import DamageDataset
from data.transforms import get_transform
import config
from torch.utils.tensorboard import SummaryWriter
import datetime
import json

from evaluation_metrics import get_coco_results, compute_coco_metrics, COCO

def log_metrics_to_tensorboard(writer, metrics, epoch):
    writer.add_scalar('mAP/AP', metrics['AP'], epoch+1)
    writer.add_scalar('mAP/AP50', metrics['AP50'], epoch+1)
    writer.add_scalar('mAP/AP75', metrics['AP75'], epoch+1)
    writer.add_scalar('mAP/AP_small', metrics['APs'], epoch+1)
    writer.add_scalar('mAP/AP_medium', metrics['APm'], epoch+1)
    writer.add_scalar('mAP/AP_large', metrics['APl'], epoch+1)

def main():
    # Create the output directories
    logs_dir = os.path.join(config.OUTPUT_DIR, config.Run, 'logs')
    checkpoints_dir = os.path.join(config.OUTPUT_DIR, config.Run, 'checkpoints')
    metrics_dir = os.path.join(config.OUTPUT_DIR, config.Run, 'metrics')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Initialize TensorBoard
    log_dir = os.path.join(logs_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to {log_dir}")

    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = DamageDataset(config.train_dir, config.train_ann, transforms=get_transform(train=True))
    val_dataset = DamageDataset(config.val_dir, config.val_ann, transforms=get_transform(train=False))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch_size=1 for evaluation
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Initialize the model
    model = get_model(config.model_name, config.NUM_CLASSES, pretrained=True, device=config.DEVICE)
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        # Train for one epoch
        train_one_epoch(model, optimizer, train_loader, device, epoch, writer, print_freq=config.PRINT_FREQ)

        # Update the learning rate
        lr_scheduler.step()

        # Compute validation loss
        val_loss = compute_validation_loss(model, val_loader, device)
        print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")

        # Log validation loss to TensorBoard
        writer.add_scalar('Loss/Validation', val_loss, epoch+1)

        # Generate and log validation visualizations
        generate_validation_visualizations(model, val_loader, device, writer, epoch, num_images=5, threshold=0.5)

        # Compute evaluation metrics on validation set
        print("Computing evaluation metrics...")
        # Load ground truth annotations
        coco_gt = COCO(config.val_ann)

        # Generate COCO-formatted predictions
        coco_results = get_coco_results(model, val_loader, device, threshold=0.5)

        # Compute metrics
        metrics = compute_coco_metrics(coco_gt, coco_results, iou_type='bbox')

        # Log metrics to TensorBoard
        log_metrics_to_tensorboard(writer, metrics, epoch)

        # Print metrics
        print("\nValidation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Save metrics to a JSON file
        metrics_path = os.path.join(metrics_dir, f'metrics_epoch_{epoch+1}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Log learning rate to TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch+1)
        print(f"Current Learning Rate: {current_lr}")

        # Save model after each epoch
        checkpoint_path = os.path.join(checkpoints_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    print("\nTraining complete.")
    writer.close()

if __name__ == "__main__":
    main()
