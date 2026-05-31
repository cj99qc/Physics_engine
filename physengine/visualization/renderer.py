from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Renderer2D:
    """
    Manages the matplotlib figure and artists for 2D particle rendering.
    Separated from RealtimeAnimator so the rendering layer can be swapped.

    Parameters
    ----------
    xlim, ylim     : axis limits
    particle_size  : scatter marker size
    figsize        : figure size in inches
    title          : optional window title
    """

    def __init__(
        self,
        xlim:          tuple[float, float] = (-10.0, 10.0),
        ylim:          tuple[float, float] = (-10.0, 10.0),
        particle_size: float = 20.0,
        figsize:       tuple[float, float] = (8.0, 8.0),
        title:         str = "Particle Simulation",
    ):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#0d1117')
        self.fig.patch.set_facecolor('#0d1117')
        self.ax.tick_params(colors='#8b949e')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#30363d')

        self._scatter = self.ax.scatter(
            [], [], s=particle_size, c='#58a6ff', zorder=3, alpha=0.9
        )
        self._time_text = self.ax.text(
            0.02, 0.97, '', transform=self.ax.transAxes,
            color='#e6edf3', fontsize=11, va='top', family='monospace'
        )
        self._trail_lines: list[Line2D] = []

    def update_particles(self, positions: np.ndarray) -> None:
        """positions: (N, 2)"""
        self._scatter.set_offsets(positions)

    def update_trails(self, trail: np.ndarray) -> None:
        """trail: (K, N, 2) — K historical positions for N particles"""
        _, N, _ = trail.shape

        while len(self._trail_lines) < N:
            line, = self.ax.plot([], [], color='#58a6ff', alpha=0.25,
                                  linewidth=0.8, zorder=2)
            self._trail_lines.append(line)

        for p in range(N):
            self._trail_lines[p].set_data(trail[:, p, 0], trail[:, p, 1])

    def update_time(self, t: float) -> None:
        self._time_text.set_text(f't = {t:.4f} s')

    def artists(self) -> list:
        """Return all artist handles for blit=True."""
        return [self._scatter, self._time_text] + self._trail_lines
