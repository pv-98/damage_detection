
import torch
import time
import datetime
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, print_freq=50):
    """
    Train the model for one epoch and log the losses to TensorBoard.
    """
    model.train()
    # Initialize loss metrics
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0

    start_time = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass the model to get the losses
        loss_dict = model(images, targets)
        #print(f'losses:{loss_dict}')
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        losses.backward()
        optimizer.step()

        # Update running loss
        running_loss += losses.item()
        running_loss_classifier += loss_dict.get('loss_classifier', 0).item()
        running_loss_box_reg += loss_dict.get('loss_box_reg', 0).item()
        running_loss_objectness += loss_dict.get('loss_objectness', 0).item()
        running_loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', 0).item()

        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / print_freq
            avg_loss_classifier = running_loss_classifier / print_freq
            avg_loss_box_reg = running_loss_box_reg / print_freq
            avg_loss_objectness = running_loss_objectness / print_freq
            avg_loss_rpn_box_reg = running_loss_rpn_box_reg / print_freq

            print(f"[Epoch {epoch+1}][Iter {i+1}/{len(data_loader)}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Loss_classifier: {avg_loss_classifier:.4f} "
                  f"Loss_box_reg: {avg_loss_box_reg:.4f} "
                  f"Loss_objectness: {avg_loss_objectness:.4f} "
                  f"Loss_rpn_box_reg: {avg_loss_rpn_box_reg:.4f}")

            # Log to TensorBoard
            current_iter = epoch * len(data_loader) + i + 1
            writer.add_scalar('Loss/Total', avg_loss, current_iter)
            writer.add_scalar('Loss/Loss_classifier', avg_loss_classifier, current_iter)
            writer.add_scalar('Loss/Loss_box_reg', avg_loss_box_reg, current_iter)
            writer.add_scalar('Loss/Loss_objectness', avg_loss_objectness, current_iter)
            writer.add_scalar('Loss/Loss_rpn_box_reg', avg_loss_rpn_box_reg, current_iter)

            # Reset running loss
            running_loss = 0.0
            running_loss_classifier = 0.0
            running_loss_box_reg = 0.0
            running_loss_objectness = 0.0
            running_loss_rpn_box_reg = 0.0

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1} training complete in {str(datetime.timedelta(seconds=int(elapsed)))}")

def compute_validation_loss(model, val_loader, device):
    """
    Compute validation loss.
    """
    model.train()

    # Set dropout and batchnorm layers to eval mode
    def set_bn_dropout_eval(m):
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.Dropout)):
            m.eval()

    model.apply(set_bn_dropout_eval)

    loss_list = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute the loss dictionary
            loss_dict = model(images, targets)

            # Sum the losses
            losses = sum(loss for loss in loss_dict.values())
            loss_list.append(losses.item())

    val_loss = sum(loss_list) / len(loss_list)
    return val_loss

def generate_validation_visualizations(model, val_loader, device, writer, epoch, num_images=5, threshold=0.5):
    """
    Get predictions and visualize.
    """
    # Set the model to evaluation mode for prediction
    model.eval()

    images_so_far = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]

            # Get model predictions
            outputs = model(images)

            for i in range(len(images)):
                if images_so_far >= num_images:
                    return  

                img = images[i].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)
                img = (img * 255).astype(np.uint8)  # Convert back to [0, 255]
                
                # Draw bounding boxes
                boxes = outputs[i]['boxes'].cpu().numpy()
                scores = outputs[i]['scores'].cpu().numpy()
                labels = outputs[i]['labels'].cpu().numpy()

                # Apply threshold
                keep = scores >= threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                for box, score, label in zip(boxes, scores, labels):
                    if label == 0:
                        continue  
                    x_min, y_min, x_max, y_max = box.astype(int)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    text = f"{CLASS_NAMES[label-1]}: {score:.2f}"
                    cv2.putText(img, text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Convert RGB to BGR for TensorBoard
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Add image to TensorBoard
                writer.add_image(f'Validation/Image_{images_so_far+1}', img, epoch+1, dataformats='HWC')

                images_so_far += 1
                if images_so_far >= num_images:
                    break  
