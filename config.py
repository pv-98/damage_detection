import os
import torch

# dataset
data_dir = './vehicle_damage_detection_dataset'
train_dir = os.path.join(data_dir, 'images', 'train')
val_dir = os.path.join(data_dir, 'images','val')
test_dir = os.path.join(data_dir, 'images', 'test')

train_ann = os.path.join(data_dir, 'annotations', 'instances_train.json')
val_ann = os.path.join(data_dir, 'annotations', 'instances_val.json')
test_ann = os.path.join(data_dir, 'annotations', 'instances_test.json')

# Hyperparameters
NUM_CLASSES = 9  
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE = 5
GAMMA = 0.1

# Model
model_name = 'faster_rcnn'

# Miscellaneous
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = 'cpu'
NUM_WORKERS = 4
PRINT_FREQ = 5  # Print training stats every N iterations
OUTPUT_DIR = './output'
Run = 'first'
MODEL_PATH = './output/model_epoch_30.pth'
