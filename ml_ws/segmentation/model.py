from pathlib import Path

import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


ROOT = next((path for path in Path(__file__).resolve().parents if (path / 'pixi.toml').is_file()), Path.cwd())


def build_model(settings, initialize=True):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if initialize and settings['pretrained'] else None
    if weights is not None:
        torch.hub.set_dir(str(ROOT / 'ml_ws/segmentation/weights'))
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=None, trainable_backbone_layers=5 if weights is not None else None, min_size=settings['min_size'], max_size=settings['max_size'],
        box_score_thresh=settings['score_threshold'], box_detections_per_img=settings['max_detections'],
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    return model


def resolve_device(name):
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; run with GPU access or explicitly use --device cpu')
    return device


def output_directory(name):
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('Experiment name must be a single folder name')
    path = ROOT / 'ml_ws/outputs/segmentation' / name
    if path.exists() and any(path.iterdir()):
        if (path / 'checkpoints/last.pt').is_file():
            raise ValueError(f'Experiment already exists: {path}. Resume with --resume {path / "checkpoints/last.pt"} or use a new --name')
        raise ValueError(f'Experiment directory already contains files: {path}. No resumable checkpoint exists; move this directory aside or use a new --name')
    return path
