

from .faster_rcnn import FasterRCNNModel
from .yolo import YOLOModel

def get_model(model_name, num_classes, pretrained=True, device=None):
    """
    To select the desired model. (It helps in scalability)

    Args:
        model_name (str): Name of the model to instantiate ('faster_rcnn' or 'yolo').
        num_classes (int): Number of classes for the head.
        pretrained (bool, optional): If True, use a pre-trained backbone (where applicable). Defaults to True.
        device (str, optional): cpu or gpu

    Returns:
         The selected model.
    """
    model_name = model_name.lower()
    if model_name == 'faster_rcnn':
        model = FasterRCNNModel(num_classes, pretrained, device).get_model()
    elif model_name == 'yolo':
        model = YOLOModel(num_classes, pretrained, device).get_model()
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'. Supported models: 'faster_rcnn', 'yolo'.")
    return model
