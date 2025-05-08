import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b5, EfficientNet_B5_Weights
from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel

class SkinEfficientNetB5(BaseModel):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=False):
        super(SkinEfficientNetB5, self).__init__()
        self.backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)

        # Unfreeze all layers if fine-tuning
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)


    @staticmethod
    def name():
        return "SkinEfficientNetB5"

    def forward(self, x):
        return self.backbone(x)
