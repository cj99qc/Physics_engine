"""
Export backends for DataRecorder:
  - NumpyExporter  → compressed .npz
  - CSVExporter    → flat .csv files
  - HDF5Exporter   → hierarchical .h5 (requires h5py)
"""
from __future__ import annotations
import csv
import numpy as np
from pathlib import Path
from .recorder import DataRecorder


class NumpyExporter:
    """
    Save all trajectory data to a single compressed .npz file.

    Load with:
        data = np.load('run.npz')
        positions = data['positions']  # (T, N, D)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def export(self, recorder: DataRecorder) -> None:
        np.savez_compressed(
            self.path,
            times=recorder.times,
            positions=recorder.positions,
            velocities=recorder.velocities,
            kinetic_energy=recorder.kinetic_energy,
            potential_energy=recorder.potential_energy,
            total_energy=recorder.total_energy,
        )
        print(f"[NumpyExporter] Saved {len(recorder)} steps to {self.path}.npz")


class CSVExporter:
    """
    Save trajectory and energy data as CSV files.

    Trajectory CSV columns: time, particle_id, x, y, [z], vx, vy, [vz]
    Energy CSV columns:     time, kinetic_energy, potential_energy, total_energy

    Parameters
    ----------
    traj_path   : path for trajectory CSV
    energy_path : optional path for energy CSV (skipped if None)
    """

    def __init__(self, traj_path: str | Path, energy_path: str | Path | None = None):
        self.traj_path   = Path(traj_path)
        self.energy_path = Path(energy_path) if energy_path else None

    def export(self, recorder: DataRecorder) -> None:
        times = recorder.times       # (T,)
        pos   = recorder.positions   # (T, N, D)
        vel   = recorder.velocities  # (T, N, D)
        T, N, D = pos.shape

        # Ensure parent directory exists
        self.traj_path.parent.mkdir(parents=True, exist_ok=True)

        dim_labels = ['x', 'y', 'z'][:D]
        vel_labels = [f'v{l}' for l in dim_labels]

        with self.traj_path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'particle_id'] + dim_labels + vel_labels)
            for t_i in range(T):
                for p_i in range(N):
                    w.writerow(
                        [times[t_i], p_i]
                        + pos[t_i, p_i].tolist()
                        + vel[t_i, p_i].tolist()
                    )

        print(f"[CSVExporter] Trajectory saved to {self.traj_path}")

        if self.energy_path is not None and recorder.record_energy:
            self.energy_path.parent.mkdir(parents=True, exist_ok=True)
            with self.energy_path.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time', 'kinetic_energy', 'potential_energy', 'total_energy'])
                for t_i in range(T):
                    w.writerow([
                        times[t_i],
                        recorder.kinetic_energy[t_i],
                        recorder.potential_energy[t_i],
                        recorder.total_energy[t_i],
                    ])
            print(f"[CSVExporter] Energy saved to {self.energy_path}")


class HDF5Exporter:
    """
    Save to HDF5 format via h5py (install with: pip install h5py).

    HDF5 layout:
        /trajectory/times       (T,)      float64
        /trajectory/positions   (T, N, D) float64
        /trajectory/velocities  (T, N, D) float64
        /energy/kinetic         (T,)      float64
        /energy/potential       (T,)      float64
        /energy/total           (T,)      float64
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def export(self, recorder: DataRecorder) -> None:
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 export. Install with: pip install h5py"
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.path, 'w') as hf:
            traj = hf.create_group('trajectory')
            traj.create_dataset('times',      data=recorder.times,      compression='gzip')
            traj.create_dataset('positions',  data=recorder.positions,  compression='gzip')
            traj.create_dataset('velocities', data=recorder.velocities, compression='gzip')
            en = hf.create_group('energy')
            en.create_dataset('kinetic',   data=recorder.kinetic_energy,  compression='gzip')
            en.create_dataset('potential', data=recorder.potential_energy, compression='gzip')
            en.create_dataset('total',     data=recorder.total_energy,     compression='gzip')

        print(f"[HDF5Exporter] Saved {len(recorder)} steps to {self.path}")
