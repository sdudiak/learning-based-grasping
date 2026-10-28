import json

import numpy as np


def instance_annotations(data, objects):
    labels = data['info']['idToLabels']
    image = np.asarray(data['data'])
    if image.ndim != 2 or not image.size:
        raise RuntimeError('Instance annotator returned no label image; check rendering/GPU availability')
    masks, ids = [], []
    instance_map = np.zeros(image.shape, dtype=np.uint32)
    for obj in objects:
        render_ids = []
        for key, value in labels.items():
            path = value if isinstance(value, str) else value.get('primPath', '')
            if path == obj['prim_path'] or path.startswith(obj['prim_path'] + '/'):
                render_ids.append(int(key))
        mask = np.isin(image, render_ids)
        if not mask.any():
            continue
        masks.append(mask)
        instance_map[mask] = obj['id']
        ids.append(obj['id'])
    return {'masks': np.asarray(masks, dtype=bool).reshape(-1, *image.shape),
            'instance_map': instance_map,
            'render_instance_map': image.astype(np.uint32),
            'instance_ids': np.asarray(ids, dtype=np.int32)}


class Capture:
    def __init__(self, resolution, save_masked_images=False):
        self.save_masked_images = save_masked_images
        import omni.replicator.core as rep
        self.rep = rep
        rep.orchestrator.set_capture_on_play(False)
        self.product = rep.create.render_product('/World/Camera', resolution)
        self.rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        self.instances = rep.AnnotatorRegistry.get_annotator('instance_segmentation', init_params={'colorize': False})
        for annotator in (self.rgb, self.instances):
            annotator.attach(self.product)

    def save(self, output, frame_id, objects, metadata):
        from PIL import Image
        self.rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=True)
        rgb = np.asarray(self.rgb.get_data())
        if rgb.ndim != 3 or rgb.shape[2] < 3 or not rgb.size:
            raise RuntimeError('RGB annotator returned no image; a working RTX GPU is required')
        instance_data = self.instances.get_data()
        annotations = instance_annotations(instance_data, objects)
        if annotations['masks'].shape[1:] != rgb.shape[:2]:
            raise RuntimeError('RGB and annotation dimensions differ')
        if not len(annotations['instance_ids']):
            raise RuntimeError('No visible object instances captured; refusing an empty ground-truth frame')
        image = output / 'images' / f'{frame_id}.png'
        truth = output / 'ground_truth' / f'{frame_id}.npz'
        Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(image)
        np.savez_compressed(truth, **annotations)
        details = {'id': frame_id, 'image': str(image.relative_to(output)),
                   'ground_truth': str(truth.relative_to(output)), 'objects': objects,
                   'visible_instance_ids': annotations['instance_ids'].tolist(),
                   'render_id_to_labels': instance_data['info']['idToLabels'], **metadata}
        if self.save_masked_images:
            from .save_masked_images import save_masked_image
            masked = output / 'masked_images' / f'{frame_id}.png'
            save_masked_image(rgb, annotations['masks'], annotations['instance_ids'], masked)
            details['masked_image'] = str(masked.relative_to(output))
        (output / 'ground_truth' / f'{frame_id}.json').write_text(json.dumps(details, indent=2) + '\n')
        return details

    def close(self):
        try:
            self.rgb.detach(self.product)
        finally:
            try:
                self.instances.detach(self.product)
            finally:
                self.product.destroy()
