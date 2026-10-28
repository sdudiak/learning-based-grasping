import math
import random

import numpy as np
import torch

from ml_ws.segmentation.metrics import aggregate, mask_iou


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def train_epoch(model, loader, optimizer, scaler, device, steps=None):
    model.train()
    steps = steps or len(loader)
    iterator = iter(loader)
    totals = {}
    for step in range(steps):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not math.isfinite(loss.item()):
            raise RuntimeError(f'Non-finite training loss: {losses}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        for key, value in {**losses, 'total': loss}.items():
            totals[key] = totals.get(key, 0.0) + value.item()
    return {key: value / steps for key, value in totals.items()}


@torch.inference_mode()
def validation_loss(model, loader, device, amp=False):
    modes = {module: module.training for module in model.modules()}
    state = rng_state()
    totals = {}
    count = 0
    try:
        # Detection models return losses in training mode; keep batch norm statistics fixed.
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            with torch.autocast(device_type=device.type, enabled=amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError(f'Non-finite validation loss: {losses}')
            for key, value in {**losses, 'total': loss}.items():
                totals[key] = totals.get(key, 0.0) + value.item() * len(images)
            count += len(images)
        return {key: value / count for key, value in totals.items()}
    finally:
        for module, training in modes.items():
            module.training = training
        restore_rng(state)


@torch.inference_mode()
def validate(model, loader, device, amp=False):
    model.eval()
    samples = []
    for images, targets in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            results = model([image.to(device) for image in images])
        for prediction, target in zip(results, targets):
            masks = prediction['masks'][:, 0].float().cpu().numpy() >= 0.5
            truth = target['masks'].numpy().astype(bool)
            samples.append({'iou': mask_iou(masks, truth), 'scores': prediction['scores'].float().cpu().numpy()})
    return aggregate(samples, 0.3, 0.5)
