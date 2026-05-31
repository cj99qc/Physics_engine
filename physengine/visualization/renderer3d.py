from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D


class Renderer3D:
    """
    Manages the matplotlib figure and artists for 3D particle rendering.
    Mirrors Renderer2D's API so it can be swapped in easily.

    Parameters
    ----------
    xlim, ylim, zlim : axis limits
    particle_size    : scatter marker size
    figsize          : figure size in inches
    title            : optional window title
    """

    def __init__(
        self,
        xlim:          tuple[float, float] = (-10.0, 10.0),
        ylim:          tuple[float, float] = (-10.0, 10.0),
        zlim:          tuple[float, float] = (-10.0, 10.0),
        particle_size: float = 20.0,
        figsize:       tuple[float, float] = (10.0, 8.0),
        title:         str = "Particle Simulation 3D",
    ):
        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.manager.set_window_title(title)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_zlim(*zlim)

        self.ax.set_facecolor('#0d1117')
        self.fig.patch.set_facecolor('#0d1117')

        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#30363d')
        self.ax.yaxis.pane.set_edgecolor('#30363d')
        self.ax.zaxis.pane.set_edgecolor('#30363d')

        self.ax.tick_params(colors='#8b949e')
        self.ax.set_xlabel('X', color='#8b949e')
        self.ax.set_ylabel('Y', color='#8b949e')
        self.ax.set_zlabel('Z', color='#8b949e')

        self._scatter = self.ax.scatter(
            [], [], [], s=particle_size, c='#58a6ff', zorder=3, alpha=0.9, depthshade=True
        )
        self._time_text = self.ax.text2D(
            0.02, 0.97, '', transform=self.fig.transFigure,
            color='#e6edf3', fontsize=11, va='top', family='monospace'
        )
        self._trail_lines: list = []

    def update_particles(self, positions: np.ndarray) -> None:
        """positions: (N, 3)"""
        self._scatter._offsets3d = (
            positions[:, 0], positions[:, 1], positions[:, 2]
        )

    def update_trails(self, trail: np.ndarray) -> None:
        """trail: (K, N, 3) -- K historical positions for N particles"""
        _, N, _ = trail.shape

        while len(self._trail_lines) < N:
            line, = self.ax.plot([], [], [], color='#58a6ff', alpha=0.25,
                                  linewidth=0.8, zorder=2)
            self._trail_lines.append(line)

        for p in range(N):
            self._trail_lines[p].set_data_3d(
                trail[:, p, 0], trail[:, p, 1], trail[:, p, 2]
            )

    def update_time(self, t: float) -> None:
        self._time_text.set_text(f't = {t:.4f} s')

    def artists(self) -> list:
        """Return all artist handles for blit=True."""
        return [self._scatter, self._time_text] + self._trail_lines
