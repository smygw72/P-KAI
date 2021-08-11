from .resnet import get_resnet


def get_model(arch, pretrained):
    if arch == 'resnet34':
        return get_resnet('34', pretrained)
    elif arch == 'resnet50':
        return get_resnet('50', pretrained)
