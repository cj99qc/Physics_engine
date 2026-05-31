from __future__ import annotations
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ..core.simulation import Simulation
from .renderer import Renderer2D


class RealtimeAnimator:
    """
    Attaches to a Simulation and renders it in real-time using
    matplotlib FuncAnimation.

    The animation loop IS the simulation loop: each frame calls
    sim.step() one or more times, then updates the scatter plot.

    Parameters
    ----------
    simulation      : Simulation — the simulation to animate
    steps_per_frame : int — sim.step() calls per rendered frame
    interval        : int — milliseconds between frames
    xlim, ylim      : axis limits
    particle_size   : scatter marker size
    trail_length    : number of historical positions drawn as trails (0 = off)
    title           : window title
    """

    def __init__(
        self,
        simulation:      Simulation,
        steps_per_frame: int = 1,
        interval:        int = 20,
        xlim:            tuple[float, float] = (-10.0, 10.0),
        ylim:            tuple[float, float] = (-10.0, 10.0),
        particle_size:   float = 20.0,
        trail_length:    int = 0,
        title:           str = "Particle Simulation",
    ):
        self.sim             = simulation
        self.steps_per_frame = steps_per_frame
        self.interval        = interval
        self.trail_length    = trail_length

        self._trail_buf: deque[np.ndarray] = deque(maxlen=max(trail_length, 1))

        self.renderer = Renderer2D(
            xlim=xlim,
            ylim=ylim,
            particle_size=particle_size,
            title=title,
        )
        self._anim: animation.FuncAnimation | None = None

    def _update(self, frame: int):
        """FuncAnimation callback — advance physics, then update artists."""
        for _ in range(self.steps_per_frame):
            self.sim.step()

        pos = self.sim.state.positions  # (N, 2)
        self._trail_buf.append(pos.copy())

        self.renderer.update_particles(pos)

        if self.trail_length > 0 and len(self._trail_buf) > 1:
            trail = np.stack(list(self._trail_buf), axis=0)  # (K, N, 2)
            self.renderer.update_trails(trail)

        self.renderer.update_time(self.sim.state.time)
        return self.renderer.artists()

    def run(
        self,
        n_frames:  int | None = None,
        save_path: str | None = None,
        fps:       int = 30,
    ) -> None:
        """
        Start the animation.

        Parameters
        ----------
        n_frames  : total frames to render (None = run indefinitely)
        save_path : if set, save to file (MP4 requires ffmpeg, GIF requires Pillow)
                    instead of opening an interactive window
        fps       : frames per second when saving to file
        """
        self._anim = animation.FuncAnimation(
            fig=self.renderer.fig,
            func=self._update,
            frames=n_frames,
            interval=self.interval,
            blit=True,
            cache_frame_data=False,
        )

        if save_path:
            ext = save_path.rsplit('.', 1)[-1].lower()
            if ext == 'gif':
                writer = animation.PillowWriter(fps=fps)
            else:
                writer = animation.FFMpegWriter(fps=fps)
            self._anim.save(save_path, writer=writer)
            print(f"[RealtimeAnimator] Saved animation to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
