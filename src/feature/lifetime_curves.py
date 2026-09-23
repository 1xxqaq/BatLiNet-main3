"""Masked early-cycle curves. No future-cycle information enters features."""
import numpy as np
import torch


def field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class LifetimeCurveExtractor:
    """Three signals (V, C-rate, Q/Qnom), two independent phase grids.

    Descriptors: charge/discharge Q/Qnom, charge/discharge seconds,
    coulombic/energy efficiency, nominal Ah, original cycle number.
    Missing measurements are represented by separate masks, never inferred
    from normalized signal values. Duplicate capacity samples are averaged.
    """
    def __init__(self, cycles=20, phase_points=256, max_capacity=1.2,
                 drop_cycles=(), current_threshold=0.01):
        if cycles < 4 or phase_points < 16 or max_capacity <= 0:
            raise ValueError('Invalid cycle count or capacity grid.')
        self.cycles, self.phase_points = cycles, phase_points
        self.grid = np.linspace(0, max_capacity, phase_points)
        self.drop_cycles = set(drop_cycles)
        self.current_threshold = current_threshold

    def __call__(self, cell):
        nominal = float(field(cell, 'nominal_capacity_in_Ah'))
        if not np.isfinite(nominal) or nominal <= 0:
            raise ValueError('Nominal capacity must be finite and positive.')
        width = 2 * self.phase_points
        curves = np.zeros((3, self.cycles, width), dtype=np.float32)
        mask = np.zeros((self.cycles, width), dtype=bool)
        desc = np.zeros((self.cycles, 8), dtype=np.float32)
        dmask = np.zeros_like(desc, dtype=bool)
        for h, cycle in enumerate(field(cell, 'cycle_data')[:self.cycles]):
            if h in self.drop_cycles:
                continue
            v = np.asarray(field(cycle, 'voltage_in_V'), dtype=float)
            current = np.asarray(field(cycle, 'current_in_A'), dtype=float)
            if v.ndim != 1 or current.shape != v.shape:
                raise ValueError(f'Malformed voltage/current arrays at cycle {h}.')
            rate = current / nominal
            times = field(cycle, 'time_in_s')
            times = np.asarray(times, dtype=float) if times is not None else None
            if times is not None and times.shape != v.shape:
                raise ValueError(f'Malformed time array at cycle {h}.')
            ends, energies = [], []
            for phase, (key, active) in enumerate([
                ('charge_capacity_in_Ah', rate > self.current_threshold),
                ('discharge_capacity_in_Ah', rate < -self.current_threshold),
            ]):
                q = np.asarray(field(cycle, key), dtype=float) / nominal
                if q.shape != v.shape:
                    raise ValueError(f'Malformed capacity array at cycle {h}.')
                valid = active & np.isfinite(q) & np.isfinite(v) & np.isfinite(rate) & (q >= 0)
                indices = np.flatnonzero(valid)
                end, energy = np.nan, np.nan
                if len(indices) >= 3:
                    qv = q[valid]
                    unique, inverse = np.unique(qv, return_inverse=True)
                    if len(unique) >= 3:
                        observed = (self.grid >= unique[0]) & (self.grid <= unique[-1])
                        start = phase * self.phase_points
                        section = slice(start, start + self.phase_points)
                        mask[h, section] = observed
                        counts = np.bincount(inverse)
                        for channel, values in enumerate((v[valid], rate[valid], qv)):
                            means = np.bincount(inverse, weights=values) / counts
                            out = np.interp(self.grid, unique, means)
                            curves[channel, h, section] = np.where(observed, out, 0)
                        end = float(unique[-1])
                        desc[h, phase], dmask[h, phase] = end, True
                    if times is not None:
                        # Integrate only adjacent active samples; never bridge rests,
                        # missing values, opposite phases, or reset timestamps.
                        dt = np.diff(times)
                        intervals = valid[:-1] & valid[1:] & np.isfinite(dt) & (dt > 0)
                        if intervals.any():
                            desc[h, 2 + phase] = dt[intervals].sum()
                            dmask[h, 2 + phase] = True
                            power = np.abs(v * current)
                            energy = float(((power[:-1] + power[1:]) * .5 * dt)[intervals].sum())
                ends.append(end)
                energies.append(energy)
            if mask[h].any():
                for k, numerator, denominator in [(4, ends[1], ends[0]), (5, energies[1], energies[0])]:
                    if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0:
                        desc[h, k], dmask[h, k] = numerator / denominator, True
                desc[h, 6], dmask[h, 6] = nominal, True
                number = field(cycle, 'cycle_number', h + 1)
                if number is not None and np.isfinite(float(number)):
                    desc[h, 7], dmask[h, 7] = float(number), True
        if not mask.any():
            raise ValueError(f'No observed early curve points: {field(cell, "cell_id", "unknown")}')
        return dict(curves=torch.from_numpy(curves), mask=torch.from_numpy(mask),
                    descriptors=torch.from_numpy(desc), descriptor_mask=torch.from_numpy(dmask))
