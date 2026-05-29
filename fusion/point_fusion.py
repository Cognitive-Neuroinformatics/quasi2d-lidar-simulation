import numpy as np

def fuse_multi_with_dt(
        pc_curr_sensor,
        ts_curr,
        prev_sensor_list,
        prev_ts_list,
        num_frames: int,
        *,
        frame_rate_hz: float = 10.0,       # Waymo Perception frame rate 10 Hz
        fixed_horizon_s=None,  
        clip_range: tuple[float, float] = (-1.0, 0.0),
    ):
        """
        Returns concatenated point cloud with an extra 5th column dt_norm.

        dt_norm = (ts - ts_curr) / T_const

        T_const is CONSTANT (not dependent on which frames are present):
        - If fixed_horizon_s is provided: T_const = fixed_horizon_s
        - Else: T_const = (num_frames - 1) / frame_rate_hz   (so oldest≈-1 when all frames exist)
        """

        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        # Select up to num_frames
        pcs = [pc_curr_sensor] + prev_sensor_list[: max(0, num_frames - 1)]
        tss = [ts_curr]        + prev_ts_list[: max(0, num_frames - 1)]

        # --- Constant normalization window ---
        if fixed_horizon_s is not None:
            T_const_us = float(fixed_horizon_s) * 1e6
        else:
            if num_frames == 1:
                T_const_us = 1.0  
            else:
                if frame_rate_hz <= 0:
                    raise ValueError("frame_rate_hz must be > 0")
                frame_period_us = 1e6 / float(frame_rate_hz)
                T_const_us = (num_frames - 1) * frame_period_us

        if T_const_us <= 0:
            T_const_us = 1.0

        lo, hi = clip_range
        out = []

        for pc, ts in zip(pcs, tss):
            if pc is None or pc.size == 0:
                continue

            dt_us = float(ts - ts_curr)               # <= 0 for history
            dt_norm = np.float32(dt_us / T_const_us)  # current=0, oldest≈-1 

            if clip_range is not None:
                dt_norm = np.float32(np.clip(dt_norm, lo, hi))

            dt_col = np.full((pc.shape[0], 1), dt_norm, dtype=np.float32)
            pc5 = np.hstack([pc.astype(np.float32, copy=False), dt_col])  # (N, 5)
            out.append(pc5)

        if len(out) == 0:
            return np.empty((0, 5), dtype=np.float32)

        return np.concatenate(out, axis=0)