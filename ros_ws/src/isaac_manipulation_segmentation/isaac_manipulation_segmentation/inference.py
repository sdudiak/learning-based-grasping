from pathlib import Path

import numpy as np
import torch

from ml_ws.segmentation.model import build_model, resolve_device


class Segmenter:
    def __init__(self, checkpoint, device, score_threshold, mask_threshold):
        checkpoint = torch.load(Path(checkpoint).expanduser(), map_location='cpu', weights_only=False)
        settings = dict(checkpoint['config']['model'], score_threshold=score_threshold)
        self.device = resolve_device(device)
        self.model = build_model(settings, initialize=False).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.mask_threshold = mask_threshold

    @torch.inference_mode()
    def predict(self, rgb):
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        image = image.to(device=self.device, dtype=torch.float32).div_(255)
        result = self.model([image])[0]
        return {
            'masks': result['masks'][:, 0].cpu().numpy() >= self.mask_threshold,
            'boxes': result['boxes'].cpu().numpy(),
            'scores': result['scores'].cpu().numpy(),
            'labels': result['labels'].cpu().numpy(),
        }
