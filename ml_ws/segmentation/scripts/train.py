import argparse
import hashlib
import json
import math
import shutil
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_ws.segmentation.dataset import PreparedDataset, collate, load_splits
from ml_ws.segmentation.engine import restore_rng, rng_state, seed_everything, train_epoch, validate, validation_loss, worker_seed
from ml_ws.segmentation.model import build_model, output_directory, resolve_device


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs/maskrcnn.toml'


def split_fingerprints(dataset):
    return {name: hashlib.sha256((dataset / 'splits' / f'{name}.json').read_bytes()).hexdigest()
            for name in ('train', 'validation', 'test')}


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def log_metrics(writer, split, metrics, epoch):
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f'{split}/{key}', value, epoch)


def main():
    parser = argparse.ArgumentParser(description='Train single-class Mask R-CNN on generated instance masks.')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--name')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--overfit-images', type=int, default=0, help='Train/evaluate on N training images without augmentation; diagnostic only')
    parser.add_argument('--steps-per-epoch', type=int, help='Repeat training batches to this number of updates per epoch')
    parser.add_argument('--resume', type=Path, help='Resume a trusted last.pt checkpoint in its original experiment directory')
    args = parser.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    checkpoint = None
    if args.resume:
        if args.resume.name != 'last.pt':
            parser.error('Resume from last.pt; use best.pt for prediction')
        if args.name:
            parser.error('--resume continues its original experiment; do not provide --name')
        if args.config.resolve() != DEFAULT_CONFIG.resolve():
            parser.error('Resume uses the checkpoint configuration; do not provide --config')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        dataset = Path(checkpoint['dataset'])
        if args.dataset and args.dataset.resolve() != dataset:
            parser.error('Resume dataset must match the checkpoint dataset')
        output = args.resume.resolve().parents[1]
        overfit_images = checkpoint['overfit_images']
        if args.overfit_images and args.overfit_images != overfit_images:
            parser.error('Cannot change the overfit subset when resuming')
        steps = checkpoint['steps_per_epoch'] if args.steps_per_epoch is None else args.steps_per_epoch
    else:
        if not args.dataset:
            parser.error('--dataset is required for a new experiment')
        config = tomllib.loads(args.config.read_text())
        dataset = args.dataset.resolve()
        output = output_directory(args.name or datetime.now(timezone.utc).strftime('maskrcnn_%Y%m%dT%H%M%S%fZ'))
        overfit_images = args.overfit_images
        steps = args.steps_per_epoch
    for key, value in (('epochs', args.epochs), ('batch_size', args.batch_size)):
        if value is not None:
            config['training'][key] = value
    training = config['training']
    if min(training['epochs'], training['batch_size'], training['lr_step_epochs']) < 1 or training['workers'] < 0 or overfit_images < 0 or (steps is not None and steps < 1):
        parser.error('Epochs, batch size, scheduler interval and steps must be positive; workers/overfit-images nonnegative')
    if not math.isfinite(training['learning_rate']) or training['learning_rate'] <= 0:
        parser.error('Learning rate must be finite and positive')
    print('Initializing training and loading dataset splits...', flush=True)
    device = resolve_device(args.device)
    seed_everything(training['seed'])
    splits = load_splits(dataset)
    fingerprints = split_fingerprints(dataset)
    if checkpoint and fingerprints != checkpoint['split_fingerprints']:
        parser.error('Split files changed since the checkpoint was saved')
    frames = splits['train']
    if overfit_images:
        frames = [frame for frame in frames if not frame.get('augmented', False)]
    if overfit_images:
        if overfit_images > len(frames):
            parser.error(f'Overfit subset exceeds {len(frames)} training images')
        frames = frames[:overfit_images]
    train_data = PreparedDataset(dataset, frames)
    evaluation_frames = frames if overfit_images else splits['validation']
    evaluation_data = PreparedDataset(dataset, evaluation_frames)
    generator = torch.Generator().manual_seed(training['seed'])
    loader = DataLoader(train_data, batch_size=training['batch_size'], shuffle=True, collate_fn=collate,
                        num_workers=training['workers'], worker_init_fn=worker_seed, generator=generator)
    evaluation_loader = DataLoader(evaluation_data, batch_size=1, shuffle=False, collate_fn=collate,
                                   num_workers=training['workers'], worker_init_fn=worker_seed)
    print(f'Loading Mask R-CNN on {device}...', flush=True)
    model = build_model(config['model'], initialize=checkpoint is None).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=training['learning_rate'], momentum=training['momentum'],
                                weight_decay=training['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, training['lr_step_epochs'], training['lr_gamma'])
    scaler = torch.amp.GradScaler('cuda', enabled=training['amp'] and device.type == 'cuda')
    start, best = 0, -1.0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        generator.set_state(checkpoint['loader_rng'])
        restore_rng(checkpoint['rng'])
        start, best = checkpoint['epoch'], checkpoint['best_score']
        if training['epochs'] <= start:
            parser.error(f'Checkpoint already completed {start} epochs; set a larger --epochs total')
    else:
        (output / 'checkpoints').mkdir(parents=True)
        shutil.copyfile(args.config, output / 'config.toml')
        (output / 'manifest.json').write_text(json.dumps({
            'dataset': str(dataset), 'model': 'maskrcnn_resnet50_fpn_v2', 'classes': ['background', 'object'],
            'evaluation_split': 'overfit_train' if overfit_images else 'validation',
            'train_frame_ids': [f['id'] for f in frames], 'split_fingerprints': fingerprints,
            'torch_version': str(torch.__version__), 'device': str(device),
        }, indent=2) + '\n')
        print('Running baseline validation...', flush=True)
        baseline = validate(model, evaluation_loader, device, scaler.is_enabled())
        baseline_loss = validation_loss(model, evaluation_loader, device, scaler.is_enabled())
        with (output / 'training_metrics.jsonl').open('w') as stream:
            stream.write(json.dumps({'epoch': 0, 'evaluation_split': 'overfit_train' if overfit_images else 'validation',
                                     'mask': baseline, 'validation_loss': baseline_loss}, allow_nan=False) + '\n')
        print(f'Baseline mask AP: {baseline["ap50_95"]}', flush=True)
    (output / 'resolved_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Experiment: {output}; training images: {len(train_data)}; evaluation images: {len(evaluation_data)}', flush=True)
    evaluation_split = 'overfit_train' if overfit_images else 'validation'
    writer = SummaryWriter(str(output / 'tensorboard'), purge_step=start + 1 if checkpoint else None)
    try:
        if checkpoint is None:
            log_metrics(writer, evaluation_split, baseline, 0)
            log_metrics(writer, f'{evaluation_split}_loss', baseline_loss, 0)
        for epoch in range(start, training['epochs']):
            print(f'Epoch {epoch + 1}/{training["epochs"]}: training...', flush=True)
            begin = time.perf_counter()
            losses = train_epoch(model, loader, optimizer, scaler, device, steps)
            print(f'Epoch {epoch + 1}: validating...', flush=True)
            mask = validate(model, evaluation_loader, device, scaler.is_enabled())
            val_losses = validation_loss(model, evaluation_loader, device, scaler.is_enabled())
            score = mask['ap50_95'] if mask['ap50_95'] is not None else 0.0
            improved = score > best
            best = max(best, score)
            writer.add_scalar('training/learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
            scheduler.step()
            record = {'epoch': epoch + 1, 'loss': losses, 'mask': mask, 'validation_loss': val_losses,
                      'evaluation_split': 'overfit_train' if overfit_images else 'validation',
                      'seconds': time.perf_counter() - begin, 'next_learning_rate': scheduler.get_last_lr()[0]}
            with (output / 'training_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'epoch': epoch + 1, 'best_score': best, 'config': config, 'dataset': str(dataset),
                     'split_fingerprints': fingerprints, 'overfit_images': overfit_images,
                     'steps_per_epoch': steps, 'rng': rng_state(), 'loader_rng': generator.get_state()}
            save_checkpoint(output / 'checkpoints/last.pt', state)
            if improved:
                save_checkpoint(output / 'checkpoints/best.pt', state)
            print(f'Epoch {epoch + 1}: loss={losses["total"]:.4f}, mask AP={score:.4f}, seconds={record["seconds"]:.1f}', flush=True)
            for key, value in losses.items():
                writer.add_scalar(f'loss/{key}', value, epoch + 1)
            log_metrics(writer, evaluation_split, mask, epoch + 1)
            log_metrics(writer, f'{evaluation_split}_loss', val_losses, epoch + 1)
            writer.add_scalar('training/epoch_seconds', record['seconds'], epoch + 1)
            writer.flush()
        best_path = output / 'checkpoints/best.pt'
        best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(best_checkpoint['model'])
        final_mask = validate(model, evaluation_loader, device, scaler.is_enabled())
        summary = {'checkpoint': str(best_path), 'epoch': best_checkpoint['epoch'],
                   'evaluation_split': evaluation_split, 'images': len(evaluation_data),
                   'diagnostic_only': bool(overfit_images), 'mask': final_mask}
        (output / 'final_validation.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(f'Best checkpoint final {evaluation_split} mask AP: {final_mask["ap50_95"]}', flush=True)
    finally:
        writer.close()



if __name__ == '__main__':
    main()
