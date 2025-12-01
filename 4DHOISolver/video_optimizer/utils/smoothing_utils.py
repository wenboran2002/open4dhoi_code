import numpy as np
import torch
def _ensure_tensor_list(series):
    tensors = []
    shapes = []
    for v in series:
        if isinstance(v, torch.Tensor):
            t = v.detach().to(dtype=torch.float32)
        else:
            t = torch.tensor(np.asarray(v), dtype=torch.float32)
        tensors.append(t)
        shapes.append(tuple(t.shape))
    return tensors, shapes


def _to_original_type_list(tensors, ref_series):
    out = []
    for t, ref in zip(tensors, ref_series):
        arr = t.detach().cpu().numpy()
        # keep original shape if ref was array-like
        try:
            ref_shape = np.asarray(ref).shape
            arr = arr.reshape(ref_shape)
        except Exception:
            pass
        # store as python lists to be JSON-friendly
        out.append(arr.tolist())
    return out


def _ema_smooth_series(tensor_list, alpha=0.5, bidirectional=True):
    if len(tensor_list) == 0:
        return []
    with torch.no_grad():
        data = torch.stack([t for t in tensor_list], dim=0)
        T = data.shape[0]
        y_fwd = data.clone()
        for t in range(1, T):
            y_fwd[t] = alpha * data[t] + (1.0 - alpha) * y_fwd[t - 1]
        if not bidirectional:
            return [y_fwd[t] for t in range(T)]
        y_bwd = data.clone()
        for t in range(T - 2, -1, -1):
            y_bwd[t] = alpha * data[t] + (1.0 - alpha) * y_bwd[t + 1]
        y = 0.5 * (y_fwd + y_bwd)
        return [y[t] for t in range(T)]


def _box_smooth_series(tensor_list, window_size: int):
    if len(tensor_list) == 0 or window_size <= 1:
        return tensor_list
    k = max(1, int(window_size))
    if k % 2 == 0:
        k += 1
    half = k // 2
    T = len(tensor_list)
    out = []
    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        vals = [tensor_list[i] for i in range(s, e)]
        out.append(torch.mean(torch.stack(vals, dim=0), dim=0))
    return out


def _gaussian_kernel(window_size: int, sigma: float, device, dtype):
    k = max(1, int(window_size))
    if k % 2 == 0:
        k += 1
    center = k // 2
    idx = torch.arange(k, device=device, dtype=dtype)
    if sigma <= 0:
        w = torch.ones(k, device=device, dtype=dtype)
    else:
        w = torch.exp(-0.5 * ((idx - center) / sigma) ** 2)
    w = w / (w.sum() + 1e-8)
    return w


def _gaussian_smooth_series(tensor_list, window_size: int, sigma: float = None):
    if len(tensor_list) == 0 or window_size <= 1:
        return tensor_list
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype
    k = max(1, int(window_size))
    if k % 2 == 0:
        k += 1
    if sigma is None:
        sigma = max(1.0, k / 3.0)
    half = k // 2
    kernel = _gaussian_kernel(k, sigma, device, dtype)
    T = len(tensor_list)
    out = []
    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        ks = half - (t - s)
        ke = ks + (e - s)
        kw = kernel[ks:ke]
        vals = [tensor_list[i] * kw[i - s] for i in range(s, e)]
        denom = kw.sum() + 1e-8
        out.append(torch.stack(vals, dim=0).sum(dim=0) / denom)
    return out


def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    R = R.to(dtype=torch.float32)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    q = q / (torch.linalg.norm(q) + 1e-8)
    return q


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / (torch.linalg.norm(q) + 1e-8)
    w, x, y, z = q[0], q[1], q[2], q[3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)]),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)]),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)])
    ])
    return R.to(dtype=torch.float32)


def _quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    q0 = q0 / (torch.linalg.norm(q0) + 1e-8)
    q1 = q1 / (torch.linalg.norm(q1) + 1e-8)
    dot = torch.dot(q0, q1)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.where(dot < 0.0, -dot, dot)
    DOT_THRESHOLD = 0.9995
    if float(dot) > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return result / (torch.linalg.norm(result) + 1e-8)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    s0 = torch.cos(theta) - dot * sin_theta / (sin_theta_0 + 1e-8)
    s1 = sin_theta / (sin_theta_0 + 1e-8)
    return s0 * q0 + s1 * q1


def _smooth_quaternion_sequence(q_list, beta=0.5, bidirectional=True):
    if len(q_list) == 0:
        return []
    qn = [q / (torch.linalg.norm(q) + 1e-8) for q in q_list]
    T = len(qn)
    q_fwd = [None] * T
    q_fwd[0] = qn[0]
    for t in range(1, T):
        q_fwd[t] = _quaternion_slerp(q_fwd[t - 1], qn[t], torch.tensor(beta, dtype=qn[t].dtype, device=qn[t].device))
    if not bidirectional:
        return q_fwd
    q_bwd = [None] * T
    q_bwd[-1] = qn[-1]
    for t in range(T - 2, -1, -1):
        q_bwd[t] = _quaternion_slerp(q_bwd[t + 1], qn[t], torch.tensor(beta, dtype=qn[t].dtype, device=qn[t].device))
    q_out = []
    for t in range(T):
        q_out.append(_quaternion_slerp(q_fwd[t], q_bwd[t], torch.tensor(0.5, dtype=qn[t].dtype, device=qn[t].device)))
    return q_out


def _box_smooth_quaternion_sequence(q_list, window_size: int):
    if len(q_list) == 0 or window_size <= 1:
        return q_list
    k = max(1, int(window_size))
    if k % 2 == 0:
        k += 1
    half = k // 2
    T = len(q_list)
    out = []
    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        q_mean = q_list[s]
        for i in range(s + 1, e):
            n = i - s + 1
            w = 1.0 / n
            q_mean = _quaternion_slerp(q_mean, q_list[i], torch.tensor(w, dtype=q_mean.dtype, device=q_mean.device))
        out.append(q_mean / (torch.linalg.norm(q_mean) + 1e-8))
    return out


def _gaussian_smooth_quaternion_sequence(q_list, window_size: int, sigma: float = None):
    if len(q_list) == 0 or window_size <= 1:
        return q_list
    k = max(1, int(window_size))
    if k % 2 == 0:
        k += 1
    if sigma is None:
        sigma = max(1.0, k / 3.0)
    device = q_list[0].device
    dtype = q_list[0].dtype
    kernel = _gaussian_kernel(k, sigma, device, dtype)
    half = k // 2
    T = len(q_list)
    out = []
    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        ks = half - (t - s)
        ke = ks + (e - s)
        kw = kernel[ks:ke]
        q_mean = q_list[s]
        w_sum = kw[0]
        for i in range(s + 1, e):
            w_i = kw[i - s]
            gamma = float(w_i / (w_sum + w_i + 1e-8))
            q_mean = _quaternion_slerp(q_mean, q_list[i], torch.tensor(gamma, dtype=dtype, device=device))
            w_sum = w_sum + w_i
        out.append(q_mean / (torch.linalg.norm(q_mean) + 1e-8))
    return out


def _extract_series_from_container(container):
    if container is None:
        return [], []
    if isinstance(container, dict):
        # sort keys numerically if possible
        try:
            keys = sorted(container.keys(), key=lambda k: int(k))
        except Exception:
            keys = sorted(container.keys())
        series = [container[k] for k in keys]
        return series, keys
    if isinstance(container, (list, tuple)):
        keys = list(range(len(container)))
        series = list(container)
        return series, keys
    # single value
    return [container], [0]


def _restore_series_to_container(container, smoothed, keys):
    if isinstance(container, dict):
        for k, v in zip(keys, smoothed):
            container[k] = v
        return container
    if isinstance(container, list):
        for i, v in zip(keys, smoothed):
            container[i] = v
        return container
    # single value
    return smoothed[0]


def lowpass_smooth_all_dict(
    data: dict,
    # alpha: float = 0.25,
    alpha: float = 0.5,
    beta_quat: float = 0.25,
    bidirectional: bool = True,
    ema_passes: int = 2,
    # window_size: int = 7,
    window_size: int = 3,
    method: str = 'ema_box',
    cutoff: float = 0.08,
    butter_order: int = 4,
    fs: float = 1.0,
):
    _ = (cutoff, butter_order, fs)  
    human = data.get('human_params_transformed', {}) if isinstance(data, dict) else {}
    obj = data.get('object_params_transformed', {}) if isinstance(data, dict) else {}

    # Human parameters smoothing
    for field in ['body_pose', 'betas', 'left_hand_pose', 'right_hand_pose']:
        if field in human and human[field] is not None:
            series_raw, keys = _extract_series_from_container(human[field])
            if len(series_raw) == 0:
                continue
            tensors, _ = _ensure_tensor_list(series_raw)
            series = tensors
            if method in ('ema', 'ema_box', 'gaussian'):
                for _pass in range(max(1, int(ema_passes))):
                    series = _ema_smooth_series(series, alpha=alpha, bidirectional=bidirectional)
                if method == 'ema_box' and window_size and window_size > 1:
                    series = _box_smooth_series(series, window_size)
                if method == 'gaussian' and window_size and window_size > 1:
                    series = _gaussian_smooth_series(series, window_size)
            smoothed_py = _to_original_type_list(series, series_raw)
            human[field] = _restore_series_to_container(human[field], smoothed_py, keys)

    # Object parameters smoothing (R as quaternion + SLERP, T as EMA/box/gaussian)
    if 'R_total' in obj and obj['R_total'] is not None:
        R_series_raw, R_keys = _extract_series_from_container(obj['R_total'])
        # convert rotation matrices to quaternions; skip Nones
        q_list = []
        valid_mask = []
        for R in R_series_raw:
            if R is None:
                q_list.append(None)
                valid_mask.append(False)
                continue
            if isinstance(R, torch.Tensor):
                Rt = R.detach().cpu().to(dtype=torch.float32)
            else:
                Rt = torch.tensor(np.asarray(R), dtype=torch.float32)
            if Rt.numel() == 9:
                Rt = Rt.view(3, 3)
            q_list.append(_rotation_matrix_to_quaternion(Rt))
            valid_mask.append(True)
        # fill missing with nearest valid to stabilize smoothing
        if any(not m for m in valid_mask):
            # forward fill then backward fill
            last_q = None
            for i in range(len(q_list)):
                if q_list[i] is None:
                    q_list[i] = last_q
                else:
                    last_q = q_list[i]
            last_q = None
            for i in range(len(q_list) - 1, -1, -1):
                if q_list[i] is None:
                    q_list[i] = last_q
                else:
                    last_q = q_list[i]
            # if still None (all were None), skip
        if len([q for q in q_list if q is not None]) > 0:
            qs = q_list
            if method in ('ema', 'ema_box', 'gaussian'):
                for _pass in range(max(1, int(ema_passes))):
                    qs = _smooth_quaternion_sequence(qs, beta=beta_quat, bidirectional=bidirectional)
                if method == 'ema_box' and window_size and window_size > 1:
                    qs = _box_smooth_quaternion_sequence(qs, window_size)
                if method == 'gaussian' and window_size and window_size > 1:
                    qs = _gaussian_smooth_quaternion_sequence(qs, window_size)
            R_smoothed = []
            for q in qs:
                if q is None:
                    R_smoothed.append(None)
                else:
                    R_smoothed.append(_quaternion_to_rotation_matrix(q).detach().cpu().numpy().tolist())
            obj['R_total'] = _restore_series_to_container(obj['R_total'], R_smoothed, R_keys)

    if 'T_total' in obj and obj['T_total'] is not None:
        T_series_raw, T_keys = _extract_series_from_container(obj['T_total'])
        if len(T_series_raw) > 0:
            # Convert possible torch tensors to CPU float32
            tensors_T = []
            for v in T_series_raw:
                if isinstance(v, torch.Tensor):
                    tensors_T.append(v.detach().cpu().to(dtype=torch.float32))
                else:
                    tensors_T.append(torch.tensor(np.asarray(v), dtype=torch.float32))
            ts = tensors_T
            if method in ('ema', 'ema_box', 'gaussian'):
                for _pass in range(max(1, int(ema_passes))):
                    ts = _ema_smooth_series(ts, alpha=alpha, bidirectional=bidirectional)
                if method == 'ema_box' and window_size and window_size > 1:
                    ts = _box_smooth_series(ts, window_size)
                if method == 'gaussian' and window_size and window_size > 1:
                    ts = _gaussian_smooth_series(ts, window_size)
            T_smoothed_py = _to_original_type_list(ts, T_series_raw)
            obj['T_total'] = _restore_series_to_container(obj['T_total'], T_smoothed_py, T_keys)

    data['human_params_transformed'] = human
    data['object_params_transformed'] = obj
    return data


