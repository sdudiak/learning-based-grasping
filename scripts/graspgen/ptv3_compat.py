import re

from grasp_gen.models.ptv3.ptv3_vanilla import PointTransformerV3Vanilla


class PointTransformerV3(PointTransformerV3Vanilla):
    def __init__(self, cls_mode=True, **kwargs):
        if not cls_mode:
            raise ValueError('The compatibility backend supports encoder-only PTv3')
        super().__init__(**kwargs)
        self.register_load_state_dict_pre_hook(self.convert_weights)

    @staticmethod
    def convert_weights(module, state_dict, prefix, *args):
        for key in list(state_dict):
            if not key.startswith(prefix):
                continue
            local = key[len(prefix):]
            renamed = local.replace('embedding.stem.', 'embedding.')
            renamed = re.sub(r'\.cpe\.0\.', '.cpe_conv.', renamed)
            renamed = re.sub(r'\.cpe\.1\.', '.cpe_linear.', renamed)
            renamed = re.sub(r'\.cpe\.2\.', '.cpe_norm.', renamed)
            renamed = re.sub(r'\.(norm1|norm2|mlp|norm)\.0\.', r'.\1.', renamed)
            value = state_dict.pop(key)
            if value.ndim == 5:
                # spconv KRSC -> flattened spatial offsets, input, output.
                value = value.permute(1, 2, 3, 4, 0).reshape(-1, value.shape[-1], value.shape[0])
            state_dict[prefix + renamed] = value
        for key, value in module.state_dict().items():
            if key.endswith('.offsets'):
                state_dict[prefix + key] = value
