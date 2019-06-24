dependencies = ['torch', 'pretorched', 'torchvision', 'torchaudio']
import pretorched


def resneti3d50(num_classes: int = 400, pretrained: str = 'moments', *args, **kwargs):
    """
    Args:
        num_classes: Number of classes of classification layer
        pretrained: Either 'moments' or 'kinetics-400'.

    Returns:
        Inflated 3D ResNet-50
    """

    return pretorched.resneti3d50(num_classes=num_classes, pretrained=pretrained, *args, **kwargs)
