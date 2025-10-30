from __future__ import annotations
from typing import Optional
import math
import time

import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointField, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

_DATATYPES = {
    PointField.INT8:    ('<i1', 1),
    PointField.UINT8:   ('<u1', 1),
    PointField.INT16:   ('<i2', 2),
    PointField.UINT16:  ('<u2', 2),
    PointField.INT32:   ('<i4', 4),
    PointField.UINT32:  ('<u4', 4),
    PointField.FLOAT32: ('<f4', 4),
    PointField.FLOAT64: ('<f8', 8),
}

def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """쿼터니언 → yaw(rad)"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def _quat_to_rot_matrix(qx, qy, qz, qw):
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),      2*(yz - wx)],
        [    2*(xz - wy),      2*(yz + wx),   1 - 2*(xx + yy)]
    ], dtype=np.float32)

def _build_struct_dtype(fields, point_step, want=('x','y','z','intensity')):
    """PointCloud2.fields에서 원하는 필드만 추출하는 structured dtype 생성 (zero-copy view)"""
    names, fmts, offs = [], [], []
    for w in want:
        f = next((f for f in fields if f.name == w), None)
        if f is None:
            # intensity가 없을 수 있음 → 일단 자리만 확보하고 나중에 0으로 채움
            continue
        fmt, _ = _DATATYPES[f.datatype]
        names.append(f.name); fmts.append(fmt); offs.append(f.offset)
    # itemsize는 원본 point_step과 동일하게 맞춰야 stride가 유지됩니다.
    return np.dtype({'names': names, 'formats': fmts, 'offsets': offs, 'itemsize': point_step})


def _transform_xyz_intensity(cloud: PointCloud2, tf: TransformStamped, target_frame: str, logger=None) -> PointCloud2:
    # 0) 전제: 대부분의 라이다 포맷은 little-endian입니다. big-endian이면 변환 필요.
    if cloud.is_bigendian:
        # 필요 시 여기서 바꿔도 되지만, 보통 센서는 little-endian
        raise ValueError("big-endian PointCloud2 not supported in fast path")

    width  = cloud.width
    height = cloud.height
    n_pts  = width * height

    # 1) 원본 버퍼를 zero-copy view로 잡기 (파이썬 루프/리스트 생성 X)
    want = ('x','y','z','intensity')
    dtype_in = _build_struct_dtype(cloud.fields, cloud.point_step, want=want)
    arr_in = np.frombuffer(cloud.data, dtype=dtype_in, count=n_pts)

    # 2) 유효 포인트 마스킹 (NaN 필터링을 파이썬이 아니라 NumPy로)
    #    intensity가 없는 경우도 고려
    has_x = 'x' in arr_in.dtype.names
    has_y = 'y' in arr_in.dtype.names
    has_z = 'z' in arr_in.dtype.names
    has_i = 'intensity' in arr_in.dtype.names

    if not (has_x and has_y and has_z):
        raise ValueError("PointCloud2 does not contain x/y/z as FLOAT32")

    x = arr_in['x'].astype(np.float32, copy=False)
    y = arr_in['y'].astype(np.float32, copy=False)
    z = arr_in['z'].astype(np.float32, copy=False)
    if has_i:
        inten = arr_in['intensity'].astype(np.float32, copy=False)
    else:
        inten = np.zeros_like(x, dtype=np.float32)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z, inten = x[mask], y[mask], z[mask], inten[mask]
    n = x.shape[0]
    if n == 0:
        out = PointCloud2()
        out.header = cloud.header
        out.header.frame_id = target_frame
        out.height = 1
        out.width = 0
        out.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        out.is_bigendian = False
        out.point_step = 16
        out.row_step = 0
        out.is_dense = True
        out.data = b""
        return out

    # 3) TF 행렬 계산
    r = tf.transform.rotation
    t = tf.transform.translation
    R = _quat_to_rot_matrix(r.x, r.y, r.z, r.w).astype(np.float32)      # (3,3)
    T = np.array([t.x, t.y, t.z], dtype=np.float32)                     # (3,)

    # 4) (N,3) @ R^T + T  (캐시/연산 효율상 이 형태가 보통 가장 빠름)
    pts = np.stack((x, y, z), axis=1)                                   # (N,3)
    pts_tf = pts @ R.T
    pts_tf += T

    # 5) 출력 버퍼 만들기: structured array → bytes (tolist 금지)
    dtype_out = np.dtype([('x','<f4'), ('y','<f4'), ('z','<f4'), ('intensity','<f4')])
    out_arr = np.empty(n, dtype=dtype_out)  # 연속 메모리
    out_arr['x'] = pts_tf[:, 0]
    out_arr['y'] = pts_tf[:, 1]
    out_arr['z'] = pts_tf[:, 2]
    out_arr['intensity'] = inten

    # 6) PointCloud2 메시지 직접 구성
    out = PointCloud2()
    out.header = cloud.header
    out.header.frame_id = target_frame
    out.height = 1
    out.width  = n
    out.fields = [
        PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    out.is_bigendian = False
    out.point_step   = 16
    out.row_step     = out.point_step * out.width
    out.is_dense     = True
    out.data         = out_arr.tobytes(order='C')  # 리스트 변환 금지

    return out