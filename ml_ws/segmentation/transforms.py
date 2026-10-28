import torch
from torchvision.transforms import v2


def make_transforms(settings, training=False):
    transforms = [v2.ToImage()]
    if training and settings['enabled']:
        transforms.extend([
            v2.RandomHorizontalFlip(settings['horizontal_flip']),
            v2.RandomVerticalFlip(settings['vertical_flip']),
            v2.ColorJitter(brightness=settings['brightness'], contrast=settings['contrast'],
                           saturation=settings['saturation'], hue=settings['hue']),
            v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=settings['blur_probability']),
        ])
    transforms.extend([v2.ToDtype(torch.float32, scale=True), v2.ToPureTensor()])
    return v2.Compose(transforms)
