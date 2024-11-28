
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

class FasterRCNNModel:
    """
    A class for faster RCNN model
    """

    def __init__(self, num_classes, pretrained=True, device=None):
        """
        Initializes the FasterRCNNModel with the number of classes according to the requirement.

        Args:
            num_classes (int): Number of classes for the classification head.
            pretrained (bool, optional): If True, use a model pre-trained on COCO. Defaults to True.
            device (str, optional): cpu or gpu
        """
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """
        Builds the Faster R-CNN model.

        Returns:
            The configured Faster R-CNN model.
        """
        # Load a pre-trained Faster R-CNN model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.pretrained)

        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Modifying the head with the custom head
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model

    def get_model(self):
        """
        Returns the configured Faster R-CNN model.

        Returns:
            The Faster R-CNN model.
        """
        return self.model

    def __repr__(self):
        """
        Gives the model configuration

        Returns:
            str: String.
        """
        return (f"{self.__class__.__name__}(num_classes={self.num_classes}, "
                f"pretrained={self.pretrained}, device={self.device})")
