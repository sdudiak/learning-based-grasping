"""Depth-point exclusion using configured collision primitives in TF link frames."""
import numpy as np

from .processing import transform_points


def validate_geometry(geometry):
    if not geometry or not isinstance(geometry, list):
        raise ValueError('Robot geometry must be a nonempty list')
    for item in geometry:
        if not item.get('link') or item.get('shape') not in ('box', 'sphere', 'cylinder', 'mesh'):
            raise ValueError('Each robot geometry needs a link and box/sphere/cylinder shape')
        shape = item['shape']
        size = np.asarray(item.get('size', []), dtype=float)
        if shape == 'mesh':
            scale = np.asarray(item.get('scale', [1, 1, 1]), dtype=float)
            if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('Invalid mesh scale')
            item['_triangles'] = load_stl(item['mesh']) * scale
        expected = {'box': 3, 'sphere': 1, 'cylinder': 2, 'mesh': 0}[shape]
        if size.shape != (expected,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError(f'Invalid {shape} size (box XYZ; sphere radius; cylinder radius,length)')
        origin = np.asarray(item.get('origin', [0, 0, 0]), dtype=float)
        quaternion = np.asarray(item.get('quaternion_xyzw', [0, 0, 0, 1]), dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all() or quaternion.shape != (4,):
            raise ValueError('Invalid geometry origin or quaternion')
        transform_points(np.zeros((1, 3)), origin, quaternion)
    return geometry


def exclusion_mask(points, geometry, transforms, tolerance=0):
    """Transforms map link names to (camera-to-link translation, quaternion XYZW)."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Robot filter tolerance must be finite and nonnegative')
    points = np.asarray(points)
    excluded = np.zeros(points.shape[:-1], dtype=bool)
    for item in geometry:
        local = transform_points(points, *transforms[item['link']])
        origin = np.asarray(item.get('origin', [0, 0, 0]))
        q = np.asarray(item.get('quaternion_xyzw', [0, 0, 0, 1]), dtype=float)
        q = q / np.linalg.norm(q)
        local = transform_points(local - origin, [0, 0, 0], [-q[0], -q[1], -q[2], q[3]])
        size = np.asarray(item.get('size', []))
        if item['shape'] == 'mesh':
            inside = mesh_exclusion(local, item['_triangles'], tolerance)
        elif item['shape'] == 'box':
            inside = (np.abs(local) <= size / 2 + tolerance).all(axis=-1)
        elif item['shape'] == 'sphere':
            inside = np.linalg.norm(local, axis=-1) <= size[0] + tolerance
        else:
            inside = ((np.linalg.norm(local[..., :2], axis=-1) <= size[0] + tolerance) &
                      (np.abs(local[..., 2]) <= size[1] / 2 + tolerance))
        excluded |= inside
    return excluded


def load_stl(path):
    from pathlib import Path
    import struct
    data = Path(path).read_bytes()
    count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else 0
    if count and len(data) == 84 + 50 * count:
        dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
        triangles = np.frombuffer(data, dtype=dtype, offset=84, count=count)['vertices'].astype(float)
    else:
        vertices = [list(map(float, line.split()[1:])) for line in data.decode('ascii').splitlines()
                    if line.strip().startswith('vertex ')]
        triangles = np.asarray(vertices, dtype=float).reshape(-1, 3, 3)
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError('STL must contain finite triangles')
    return triangles


def mesh_exclusion(points, triangles, tolerance):
    shape = points.shape[:-1]
    points = np.asarray(points).reshape(-1, 3)
    result = np.zeros(len(points), dtype=bool)
    low, high = triangles.min(axis=(0, 1)) - tolerance, triangles.max(axis=(0, 1)) + tolerance
    candidate_indices = np.flatnonzero(((points >= low) & (points <= high)).all(axis=1))
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = b - a, c - a
    normal = np.cross(edge1, edge2)
    normal_squared = np.sum(normal * normal, axis=1)
    nondegenerate = normal_squared > 1e-20
    a, b, c, edge1, edge2, normal, normal_squared = [value[nondegenerate] for value in
                                                  (a, b, c, edge1, edge2, normal, normal_squared)]
    if not len(a):
        raise ValueError('Collision mesh contains only degenerate triangles')
    direction = np.array([1., .38196601125, .17320508076])
    h = np.cross(direction, edge2)
    determinant = np.sum(edge1 * h, axis=1)
    usable = np.abs(determinant) > 1e-12
    inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=usable)
    d00, d01, d11 = [np.sum(x * y, axis=1) for x, y in ((edge1, edge1), (edge1, edge2), (edge2, edge2))]
    denominator = d00 * d11 - d01 ** 2
    for start in range(0, len(candidate_indices), 128):
        indices = candidate_indices[start:start + 128]
        relative = points[indices, None, :] - a
        u = np.sum(relative * h, axis=-1) * inverse
        q = np.cross(relative, edge1)
        v = np.sum(q * direction, axis=-1) * inverse
        t = np.sum(q * edge2, axis=-1) * inverse
        intersections = usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-10)
        inside = intersections.sum(axis=1) % 2 == 1
        d20, d21 = np.sum(relative * edge1, axis=-1), np.sum(relative * edge2, axis=-1)
        bary_v = (d11 * d20 - d01 * d21) / denominator
        bary_w = (d00 * d21 - d01 * d20) / denominator
        projected_inside = (bary_v >= 0) & (bary_w >= 0) & (bary_v + bary_w <= 1)
        plane_distance = np.sum(relative * normal, axis=-1) ** 2 / normal_squared
        minimum = np.where(projected_inside, plane_distance, np.inf)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = second - first
            length_squared = np.sum(edge * edge, axis=-1)
            offset = points[indices, None, :] - first
            fraction = np.clip(np.sum(offset * edge, axis=-1) / length_squared, 0, 1)
            distance = np.sum((offset - fraction[:, :, None] * edge) ** 2, axis=-1)
            minimum = np.minimum(minimum, distance)
        result[indices] = inside | (minimum.min(axis=1) <= tolerance ** 2 + 1e-18)
    return result.reshape(shape)
