"""
Interactive Physics Engine Web UI
Launch with: streamlit run web_ui.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from physengine import (
    ParticleSystem, Simulation, DataRecorder,
    UniformGravity, NBodyGravity, SpringForce, LinearDrag,
    get_integrator,
    keplerian_to_cartesian, hohmann_transfer, orbital_period,
    SPHDensityComputer, SPHPressureForce, SPHViscosityForce,
    BoxBoundaryForce, SPHParticleInitializer,
)

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Physics Engine Lab",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium dark theme ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background-color: #0d1117; }

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
    color: #e6edf3 !important;
}

.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 4px 0;
}
.metric-card h4 { margin: 0 0 4px 0; font-size: 13px; color: #8b949e !important; font-weight: 500; }
.metric-card p { margin: 0; font-size: 22px; font-weight: 600; }

.scenario-header {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.scenario-header h2 { color: white !important; margin: 0 0 4px 0; }
.scenario-header p { color: rgba(255,255,255,0.8) !important; margin: 0; font-size: 14px; }

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🧪 Physics Engine Lab")
st.sidebar.markdown("---")

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Free Fall (2D)", "Spring Pendulum (2D)", "N-Body Gravity (2D)", "Orbital Transfer (2D)", "3-Body Orbit (3D)", "SPH Dam Break (2D)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Simulation Settings")

integrator_name = st.sidebar.selectbox("Integrator", ["rk4", "verlet", "leapfrog"], index=0)
dt = st.sidebar.slider("Timestep (dt)", 0.001, 0.05, 0.01, 0.001, format="%.3f")
n_steps = st.sidebar.slider("Steps", 100, 10000, 2000, 100)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Scenario Parameters")

# ── Scenario-specific parameters ─────────────────────────────────────────

if scenario == "Free Fall (2D)":
    gravity = st.sidebar.slider("Gravity (m/s^2)", 1.0, 30.0, 9.81, 0.1)
    drag_b = st.sidebar.slider("Drag coefficient", 0.0, 2.0, 0.1, 0.01)
    init_height = st.sidebar.slider("Initial height", 1.0, 50.0, 10.0, 0.5)
    init_vx = st.sidebar.slider("Initial horizontal velocity", 0.0, 20.0, 3.0, 0.5)

elif scenario == "Spring Pendulum (2D)":
    gravity = st.sidebar.slider("Gravity (m/s^2)", 1.0, 30.0, 9.81, 0.1)
    spring_k = st.sidebar.slider("Spring constant (k)", 5.0, 200.0, 50.0, 5.0)
    rest_length = st.sidebar.slider("Rest length (L0)", 0.5, 5.0, 3.0, 0.1)
    bob_vx = st.sidebar.slider("Bob initial vx", 0.0, 5.0, 1.5, 0.1)

elif scenario == "N-Body Gravity (2D)":
    G_val = st.sidebar.slider("Gravitational constant G", 0.1, 10.0, 1.0, 0.1)
    n_bodies = st.sidebar.slider("Number of bodies", 2, 8, 3, 1)
    softening = st.sidebar.slider("Softening epsilon", 0.001, 0.5, 0.05, 0.001, format="%.3f")

elif scenario == "Orbital Transfer (2D)":
    r_inner = st.sidebar.slider("Inner orbit radius", 0.5, 3.0, 1.0, 0.1)
    r_outer = st.sidebar.slider("Outer orbit radius", 2.0, 10.0, 4.0, 0.1)

elif scenario == "3-Body Orbit (3D)":
    G_val = st.sidebar.slider("Gravitational constant G", 0.1, 10.0, 1.0, 0.1)
    inclination_deg = st.sidebar.slider("Orbit 3 inclination (deg)", 0, 90, 30, 5)
    softening = st.sidebar.slider("Softening epsilon", 0.001, 0.5, 0.05, 0.001, format="%.3f")

elif scenario == "SPH Dam Break (2D)":
    sph_gravity = st.sidebar.slider("Gravity (m/s^2)", 1.0, 30.0, 9.81, 0.5)
    sph_viscosity = st.sidebar.slider("Viscosity (mu)", 0.5, 20.0, 5.0, 0.5)
    sph_pressure_k = st.sidebar.slider("Pressure stiffness (k)", 500.0, 5000.0, 2000.0, 100.0)
    sph_spacing = st.sidebar.slider("Particle spacing", 0.04, 0.12, 0.08, 0.01, format="%.2f")
    sph_steps = st.sidebar.slider("SPH steps", 100, 1000, 500, 50)

# ── Run button ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
run_clicked = st.sidebar.button("🚀 Run Simulation", use_container_width=True, type="primary")

# ── Build & run simulation ───────────────────────────────────────────────

def build_and_run():
    integrator = get_integrator(integrator_name)

    if scenario == "Free Fall (2D)":
        ps = ParticleSystem(dim=2)
        ps.add(mass=1.0, position=[0.0, init_height], velocity=[init_vx, 0.0], tag="ball")
        state = ps.build()
        recorder = DataRecorder(record_every=1, record_energy=True)
        sim = Simulation(state=state, integrator=integrator, dt=dt, recorder=recorder)
        sim.add_force(UniformGravity(g=gravity))
        if drag_b > 0:
            sim.add_force(LinearDrag(b=drag_b))
        sim.run(n_steps=n_steps)
        return sim, recorder, 2

    elif scenario == "Spring Pendulum (2D)":
        ps = ParticleSystem(dim=2)
        ps.add(mass=1e9, position=[0.0, 0.0], velocity=[0.0, 0.0], tag="anchor")
        ps.add(mass=1.0, position=[0.0, -rest_length], velocity=[bob_vx, 0.0], tag="bob")
        state = ps.build()
        recorder = DataRecorder(record_every=2, record_energy=True)
        sim = (
            Simulation(state=state, integrator=integrator, dt=dt, recorder=recorder)
            .add_force(UniformGravity(g=gravity))
            .add_force(SpringForce(i=0, j=1, k=spring_k, L0=rest_length))
        )
        sim.run(n_steps=n_steps)
        return sim, recorder, 2

    elif scenario == "N-Body Gravity (2D)":
        ps = ParticleSystem(dim=2)
        rng = np.random.default_rng(42)
        for idx in range(n_bodies):
            angle = 2.0 * np.pi * idx / n_bodies
            r = 2.0
            pos = [r * np.cos(angle), r * np.sin(angle)]
            vel = [-0.3 * np.sin(angle), 0.3 * np.cos(angle)]
            ps.add(mass=1.0, position=pos, velocity=vel, tag=f"body_{idx}")
        state = ps.build()
        recorder = DataRecorder(record_every=2, record_energy=True)
        sim = (
            Simulation(state=state, integrator=integrator, dt=dt, recorder=recorder)
            .add_force(NBodyGravity(G=G_val, epsilon=softening))
        )
        sim.run(n_steps=n_steps)
        return sim, recorder, 2

    elif scenario == "Orbital Transfer (2D)":
        mu = 1.0
        pos_ijk, vel_ijk = keplerian_to_cartesian(a=r_inner, e=0.0, i=0.0, omega=0.0, Omega=0.0, nu=0.0, mu=mu)
        dv1, dv2, tof = hohmann_transfer(r_inner, r_outer, mu)

        ps = ParticleSystem(dim=2)
        ps.add(mass=1.0, position=[0.0, 0.0], velocity=[0.0, 0.0], tag="central_body")
        ps.add(mass=1e-6, position=pos_ijk[:2].tolist(), velocity=vel_ijk[:2].tolist(), tag="spacecraft")
        state = ps.build()

        T_outer = orbital_period(r_outer, mu)
        total_time = tof + T_outer
        actual_steps = int(total_time / dt)

        recorder = DataRecorder(record_every=2, record_energy=False)
        sim = (
            Simulation(state=state, integrator=integrator, dt=dt, recorder=recorder)
            .add_force(NBodyGravity(G=1.0, epsilon=1e-8))
        )
        sim.schedule_event(time=0.0, callback=lambda s: s.state.velocities.__setitem__(1, s.state.velocities[1] + np.array([0.0, dv1])))
        sim.schedule_event(time=tof, callback=lambda s: s.state.velocities.__setitem__(1, s.state.velocities[1] + np.array([0.0, -dv2])))
        sim.run(n_steps=actual_steps)
        return sim, recorder, 2

    elif scenario == "3-Body Orbit (3D)":
        ps = ParticleSystem(dim=3)
        mu = 1.0
        inc_rad = np.radians(inclination_deg)

        # Body 1: circular in XY plane
        pos1, vel1 = keplerian_to_cartesian(a=2.0, e=0.0, i=0.0, omega=0.0, Omega=0.0, nu=0.0, mu=mu)
        ps.add(mass=0.01, position=pos1.tolist(), velocity=vel1.tolist(), tag="orbit_1")

        # Body 2: circular in XY plane, opposite phase
        pos2, vel2 = keplerian_to_cartesian(a=3.0, e=0.1, i=0.0, omega=0.0, Omega=0.0, nu=np.pi, mu=mu)
        ps.add(mass=0.01, position=pos2.tolist(), velocity=vel2.tolist(), tag="orbit_2")

        # Body 3: inclined orbit
        pos3, vel3 = keplerian_to_cartesian(a=2.5, e=0.05, i=inc_rad, omega=0.0, Omega=0.0, nu=np.pi/2, mu=mu)
        ps.add(mass=0.01, position=pos3.tolist(), velocity=vel3.tolist(), tag="orbit_3")

        # Central body
        ps.add(mass=1.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], tag="central")

        state = ps.build()
        recorder = DataRecorder(record_every=2, record_energy=True)
        sim = (
            Simulation(state=state, integrator=integrator, dt=dt, recorder=recorder)
            .add_force(NBodyGravity(G=G_val, epsilon=softening))
        )
        sim.run(n_steps=n_steps)
        return sim, recorder, 3

    elif scenario == "SPH Dam Break (2D)":
        h = sph_spacing * 2
        fluid_region = ((0.02, 0.3), (0.02, 0.6))
        box = ((0.0, 1.0), (0.0, 1.0))

        positions, particle_mass = SPHParticleInitializer.fill_box(
            region=fluid_region, spacing=sph_spacing, rho_0=1000.0
        )
        N_fluid = positions.shape[0]

        ps = ParticleSystem(dim=2)
        for i in range(N_fluid):
            ps.add(mass=particle_mass, position=positions[i].tolist(), velocity=[0.0, 0.0])
        state = ps.build()

        density = SPHDensityComputer(h=h, rest_rho=1000.0)
        recorder = DataRecorder(record_every=5, record_energy=False)

        sim = (
            Simulation(state=state, integrator=get_integrator("verlet"), dt=0.001, recorder=recorder)
            .add_force(SPHPressureForce(density, k=sph_pressure_k))
            .add_force(SPHViscosityForce(density, mu=sph_viscosity))
            .add_force(UniformGravity(g=sph_gravity, direction=[0.0, -1.0]))
            .add_force(BoxBoundaryForce(bounds=box, stiffness=20000.0, margin=sph_spacing))
        )
        sim.run(n_steps=sph_steps)
        return sim, recorder, 2


# ── Main content area ────────────────────────────────────────────────────
scenario_descriptions = {
    "Free Fall (2D)": "A ball launched sideways under uniform gravity with optional linear drag.",
    "Spring Pendulum (2D)": "A heavy anchor connected to a light bob by a spring, with gravity pulling the bob.",
    "N-Body Gravity (2D)": "Multiple bodies interacting under mutual gravitational attraction.",
    "Orbital Transfer (2D)": "A spacecraft performing a Hohmann transfer between circular orbits.",
    "3-Body Orbit (3D)": "Three test particles orbiting a central mass, one with inclination.",
    "SPH Dam Break (2D)": "Fluid column collapse using Smoothed Particle Hydrodynamics (Navier-Stokes).",
}

st.markdown(f"""
<div class="scenario-header">
    <h2>🔬 {scenario}</h2>
    <p>{scenario_descriptions[scenario]}</p>
</div>
""", unsafe_allow_html=True)

if run_clicked:
    with st.spinner("Running simulation..."):
        sim, recorder, dims = build_and_run()

    positions = recorder.positions   # (T, N, D)
    times = recorder.times           # (T,)
    T_steps, N_particles, D = positions.shape

    # ── Metrics row ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sim Time", f"{sim.time:.3f} s")
    with col2:
        st.metric("Steps Recorded", f"{T_steps}")
    with col3:
        st.metric("Particles", f"{N_particles}")
    with col4:
        st.metric("Dimensions", f"{D}D")

    st.markdown("---")

    # ── Plot layout ──────────────────────────────────────────────────
    plot_col, info_col = st.columns([3, 1])

    with plot_col:
        st.markdown("### 📈 Trajectory")

        if dims == 3:
            fig = go.Figure()
            colors = ['#58a6ff', '#f6c90e', '#ff6b6b', '#7ee787', '#ff9f43', '#bc8cff', '#f778ba', '#79c0ff']
            for p in range(N_particles):
                traj = positions[:, p, :]
                fig.add_trace(go.Scatter3d(
                    x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                    mode='lines',
                    line=dict(color=colors[p % len(colors)], width=3),
                    name=f'Particle {p}',
                ))
                # Start marker
                fig.add_trace(go.Scatter3d(
                    x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
                    mode='markers',
                    marker=dict(color=colors[p % len(colors)], size=6),
                    showlegend=False,
                ))
            fig.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor='#0d1117', gridcolor='#30363d', color='#8b949e'),
                    yaxis=dict(backgroundcolor='#0d1117', gridcolor='#30363d', color='#8b949e'),
                    zaxis=dict(backgroundcolor='#0d1117', gridcolor='#30363d', color='#8b949e'),
                    bgcolor='#0d1117',
                    aspectmode='data',
                ),
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font=dict(color='#e6edf3'),
                legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
                margin=dict(l=0, r=0, t=30, b=0),
                height=550,
            )
        else:
            fig = go.Figure()
            colors = ['#58a6ff', '#f6c90e', '#ff6b6b', '#7ee787', '#ff9f43', '#bc8cff', '#f778ba', '#79c0ff']
            for p in range(N_particles):
                traj = positions[:, p, :]
                fig.add_trace(go.Scatter(
                    x=traj[:, 0], y=traj[:, 1],
                    mode='lines',
                    line=dict(color=colors[p % len(colors)], width=2),
                    name=f'Particle {p}',
                ))
                fig.add_trace(go.Scatter(
                    x=[traj[0, 0]], y=[traj[0, 1]],
                    mode='markers',
                    marker=dict(color=colors[p % len(colors)], size=8, symbol='circle'),
                    showlegend=False,
                ))
            fig.update_layout(
                xaxis=dict(scaleanchor='y', scaleratio=1, gridcolor='#30363d', color='#8b949e', zerolinecolor='#30363d'),
                yaxis=dict(gridcolor='#30363d', color='#8b949e', zerolinecolor='#30363d'),
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font=dict(color='#e6edf3'),
                legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
                margin=dict(l=40, r=20, t=30, b=40),
                height=550,
            )

        st.plotly_chart(fig, use_container_width=True)

    with info_col:
        st.markdown("### 📊 Final State")
        for p in range(min(N_particles, 6)):
            pos_final = positions[-1, p, :]
            pos_str = ", ".join(f"{v:.3f}" for v in pos_final)
            st.markdown(f"""
<div class="metric-card">
    <h4>Particle {p}</h4>
    <p style="font-size: 14px;">({pos_str})</p>
</div>
""", unsafe_allow_html=True)

    # ── Energy plot ──────────────────────────────────────────────────
    if recorder.record_energy and len(recorder._ke) > 0:
        st.markdown("---")
        st.markdown("### ⚡ Energy Over Time")

        ke = recorder.kinetic_energy
        pe = recorder.potential_energy
        te = recorder.total_energy

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=times, y=ke, mode='lines', name='Kinetic', line=dict(color='#58a6ff', width=2)))
        fig_e.add_trace(go.Scatter(x=times, y=pe, mode='lines', name='Potential', line=dict(color='#ff6b6b', width=2)))
        fig_e.add_trace(go.Scatter(x=times, y=te, mode='lines', name='Total', line=dict(color='#f6c90e', width=2.5)))
        fig_e.update_layout(
            xaxis=dict(title='Time (s)', gridcolor='#30363d', color='#8b949e', zerolinecolor='#30363d'),
            yaxis=dict(title='Energy (J)', gridcolor='#30363d', color='#8b949e', zerolinecolor='#30363d'),
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            font=dict(color='#e6edf3'),
            legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
            margin=dict(l=40, r=20, t=30, b=40),
            height=350,
        )
        st.plotly_chart(fig_e, use_container_width=True)

        # Energy drift metric
        drift = abs(te[-1] - te[0])
        st.markdown(f"""
<div class="metric-card" style="max-width: 300px;">
    <h4>Energy Drift (|E_final - E_initial|)</h4>
    <p>{drift:.6e} J</p>
</div>
""", unsafe_allow_html=True)

else:
    # Default landing page
    st.markdown("""
    <div style="text-align: center; padding: 80px 0;">
        <p style="font-size: 64px; margin-bottom: 16px;">🚀</p>
        <h2 style="color: #e6edf3 !important;">Select a scenario and hit Run</h2>
        <p style="color: #8b949e !important; font-size: 16px;">
            Tweak the parameters in the sidebar, then press <b>Run Simulation</b> to see the results.
        </p>
    </div>
    """, unsafe_allow_html=True)
