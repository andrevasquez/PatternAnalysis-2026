import torch.nn as nn
NUM_CLASSES = 6

class UNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

class ResidualUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

class IsenseeUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

def get_model(model_name):
    if model_name == "baseline":
        return UNet(num_classes=NUM_CLASSES)
    elif model_name == "residual":
        return ResidualUNet(num_classes=NUM_CLASSES)
    elif model_name == "improved":
        return IsenseeUNet(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model name: {model_name}")