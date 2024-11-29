
def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.
    """
    return tuple(zip(*batch))
