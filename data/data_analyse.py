# This script is not part of the pipeline.

from pycocotools.coco import COCO
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Change path , will not work with this.
annotation_file = './vehicle_damage_detection_dataset/annotations/instances_train.json'

# Initialize COCO API
coco = COCO(annotation_file)

# Load categories
categories = coco.loadCats(coco.getCatIds())
category_names = [category['name'] for category in categories]
print("Categories:", category_names)

# Load images and annotations
image_ids = coco.getImgIds()
images = coco.loadImgs(image_ids)
annotations = coco.loadAnns(coco.getAnnIds())

num_images = len(images)
num_annotations = len(annotations)
print(f"Number of images: {num_images}")
print(f"Number of annotations: {num_annotations}")
num_classes = len(categories)
print(f"Number of classes: {num_classes}")

# Create a DataFrame for annotations
df_annotations = pd.DataFrame(annotations)
df_annotations['category_id'] = df_annotations['category_id'].astype(int)

# Map category IDs to names
category_id_to_name = {category['id']: category['name'] for category in categories}
df_annotations['category_name'] = df_annotations['category_id'].map(category_id_to_name)

# Count instances per class
class_counts = df_annotations['category_name'].value_counts()

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Class Distribution')
plt.xlabel('Damage Classes')
plt.ylabel('Number of Instances')
plt.xticks(rotation=45)
plt.show()

# Extract width and height from bounding boxes
df_annotations['bbox_width'] = df_annotations['bbox'].apply(lambda x: x[2])
df_annotations['bbox_height'] = df_annotations['bbox'].apply(lambda x: x[3])
df_annotations['bbox_area'] = df_annotations['bbox_width'] * df_annotations['bbox_height']

# Plot distribution of bounding box areas
plt.figure(figsize=(10,6))
sns.histplot(df_annotations['bbox_area'], bins=50, kde=True)
plt.title('Distribution of Bounding Box Areas')
plt.xlabel('Area (pixels)')
plt.ylabel('Frequency')
plt.show()

# Plot aspect ratios
df_annotations['aspect_ratio'] = df_annotations['bbox_width'] / df_annotations['bbox_height']

plt.figure(figsize=(10,6))
sns.histplot(df_annotations['aspect_ratio'], bins=50, kde=True)
plt.title('Distribution of Bounding Box Aspect Ratios')
plt.xlabel('Aspect Ratio (Width / Height)')
plt.ylabel('Frequency')
plt.show()


df_images = pd.DataFrame(images)
df_annotations = df_annotations.merge(df_images[['id', 'width', 'height']], left_on='image_id', right_on='id', suffixes=('', '_image'))

# Calculate the center of bounding boxes
df_annotations['bbox_center_x'] = df_annotations['bbox'].apply(lambda x: x[0] + x[2]/2)
df_annotations['bbox_center_y'] = df_annotations['bbox'].apply(lambda x: x[1] + x[3]/2)

# Normalize centers
df_annotations['bbox_center_x_norm'] = df_annotations['bbox_center_x'] / df_annotations['width']
df_annotations['bbox_center_y_norm'] = df_annotations['bbox_center_y'] / df_annotations['height']

# Plot heatmap of damage locations
plt.figure(figsize=(8,6))
sns.kdeplot(
    x=df_annotations['bbox_center_x_norm'],
    y=df_annotations['bbox_center_y_norm'],
    cmap="Reds",
    shade=True,
    bw_adjust=.5
)
plt.title('Heatmap of Damage Locations on Vehicles')
plt.xlabel('Normalized X Coordinate')
plt.ylabel('Normalized Y Coordinate')
plt.show()


pivot = pd.crosstab(df_annotations['image_id'], df_annotations['category_name'])

# Compute co-occurrence matrix
co_occurrence = pivot.T.dot(pivot)

# Plot heatmap
plt.figure(figsize=(12,10))
sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='Blues')
plt.title('Co-occurrence Matrix of Damage Classes')
plt.xlabel('Damage Classes')
plt.ylabel('Damage Classes')
plt.show()



# Count annotations per image
annotations_per_image = df_annotations.groupby('image_id').size()

# Plot distribution
plt.figure(figsize=(10,6))
sns.histplot(annotations_per_image, bins=range(1, annotations_per_image.max()+2), discrete=True, kde=False)
plt.title('Number of Annotations per Image')
plt.xlabel('Number of Annotations')
plt.ylabel('Number of Images')
plt.show()






