"""
ATLAS - intermediate shaft design of a two-stage spur gearbox.
(Automated Torque and Load Analysis System)

Design of Machine Elements course project, Semester VI, Mechanical Engineering.

The intermediate shaft of a two-stage reduction gearbox carries the driven
gear of stage 1 and the driving pinion of stage 2 between two ball bearings.
This tool follows the standard design sequence from Bhandari:

  1. torque on the shaft from power and speed
  2. tangential and radial tooth loads at each mesh
  3. loads resolved into vertical and horizontal planes; SFD and BMD in each,
     resultant bending moment M = sqrt(Mv^2 + Mh^2)
  4. shaft diameter by the ASME code equation with shock factors Kb, Kt
  5. rigidity check: deflection at the gears and slope at the bearings
     (this usually governs, not strength)
  6. bearing selection from required dynamic load capacity
  7. a sweep over gear positions to find the placement that minimises the
     peak bending moment

All beam results come from closed-form simply-supported point-load formulas
(superposition), validated in section 8 against PL/4, PL^3/48EI and PL^2/16EI
for a central load. Pure numpy - no scipy dependency.

Units: N, mm, MPa throughout (torque shown in N m for readability).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="ATLAS - Gearbox Intermediate Shaft Design",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Plot styling is fully specified and every st.plotly_chart call passes
# theme=None, so figures render the same in light or dark Streamlit themes.
INK = "#26303a"
BLUE = "#1f4e79"   # vertical plane
RED = "#c0392b"    # horizontal plane
PLOT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(size=12, color=INK),
    margin=dict(t=40, b=30, l=55, r=20),
)
AXIS_STYLE = dict(showline=True, linecolor="#555555", ticks="outside",
                  tickcolor="#555555", gridcolor="#e8e8e8", zeroline=False)
LEGEND_BOX = dict(font=dict(color=INK), bgcolor="rgba(255,255,255,0.85)",
                  bordercolor="#cccccc", borderwidth=1)

# ---------------------------------------------------------------
# Constants and catalogue data
# ---------------------------------------------------------------
E_STEEL = 207000.0          # MPa
SLOPE_LIMIT = 0.001         # rad, guideline for deep-groove ball bearings
# permissible transverse deflection at a spur gear mesh: 0.01 x module

STD_DIAMETERS = [25, 28, 30, 32, 35, 40, 45, 50, 55, 60, 63, 70, 80]

MATERIALS = {
    "45C8  (Sut 630, Syt 380)": (630.0, 380.0),
    "40C8  (Sut 580, Syt 330)": (580.0, 330.0),
    "50C4  (Sut 660, Syt 460)": (660.0, 460.0),
    "Custom": None,
}

# Representative deep-groove ball bearing dynamic capacities C (kN),
# 62 and 63 series, from a standard catalogue. bore: [(designation, C), ...]
BEARING_TABLE = {
    25: [("6205", 14.0), ("6305", 22.5)],
    30: [("6206", 19.5), ("6306", 28.1)],
    35: [("6207", 25.5), ("6307", 33.2)],
    40: [("6208", 30.7), ("6308", 41.0)],
    45: [("6209", 33.2), ("6309", 52.7)],
    50: [("6210", 35.1), ("6310", 61.8)],
    55: [("6211", 43.6), ("6311", 71.5)],
    60: [("6212", 52.7), ("6312", 81.9)],
    65: [("6213", 58.5), ("6313", 97.5)],
    70: [("6214", 63.7), ("6314", 104.0)],
}


# ---------------------------------------------------------------
# Mechanics core
# ---------------------------------------------------------------
def resolve_mesh_force(Ft, Fr, theta_deg, reverse_ft):
    """Resolve one mesh's tooth loads into shaft x (horizontal) and
    y (vertical) components.

    theta_deg is the angular position of the MATING gear's centre around the
    shaft, measured CCW from +x. The radial (separating) force always acts
    along the line of centres, pushing this shaft away from the mating gear,
    i.e. along -(cos t, sin t). The tangential force is perpendicular to the
    line of centres; its sense depends on rotation direction and on whether
    this gear is driven or driving, so it is exposed as a reverse toggle.
    """
    t = np.radians(theta_deg)
    s = -1.0 if reverse_ft else 1.0
    fx = -Fr * np.cos(t) + s * Ft * (-np.sin(t))
    fy = -Fr * np.sin(t) + s * Ft * (np.cos(t))
    return fx, fy


def beam_response(loads, L, xs):
    """Simply supported beam, point loads [(pos, P), ...] (signed).
    Returns V(xs), M(xs) and the two reactions, from statics."""
    RB = sum(P * p for p, P in loads) / L
    RA = sum(P for _, P in loads) - RB
    V = np.full_like(xs, RA, dtype=float)
    M = RA * xs
    for p, P in loads:
        past = xs > p
        V[past] -= P
        M[past] -= P * (xs[past] - p)
    return V, M, RA, RB


def beam_deflection(loads, L, EI, xs):
    """Deflection by superposition of the closed-form single point-load
    solution (any strength-of-materials text):
        x <= a:  y = P b x (L^2 - b^2 - x^2) / (6 L EI),  b = L - a
        x >= a:  y = P a (L - x)(2Lx - a^2 - x^2) / (6 L EI)
    End slopes: thA = P b (L^2 - b^2)/(6 L EI), thB = P a (L^2 - a^2)/(6 L EI).
    Returns y(xs) and the two end slopes."""
    y = np.zeros_like(xs, dtype=float)
    thA = thB = 0.0
    for p, P in loads:
        bb = L - p
        left = xs <= p
        y[left] += P * bb * xs[left] * (L**2 - bb**2 - xs[left]**2) / (6 * L * EI)
        xr = xs[~left]
        y[~left] += P * p * (L - xr) * (2 * L * xr - p**2 - xr**2) / (6 * L * EI)
        thA += P * bb * (L**2 - bb**2) / (6 * L * EI)
        thB += P * p * (L**2 - p**2) / (6 * L * EI)
    return y, thA, thB


def asme_diameter(M, T, Kb, Kt, tau_allow):
    """ASME code equation for a solid transmission shaft:
        d^3 = 16 / (pi tau) * sqrt((Kb M)^2 + (Kt T)^2)"""
    return (16.0 / (np.pi * tau_allow) * np.hypot(Kb * M, Kt * T)) ** (1 / 3)


@st.cache_data
def placement_sweep(L, fx2, fy2, fx3, fy3, margin, gap, step=5.0):
    """Sweep gear positions (a, b) and return the peak resultant bending
    moment for every feasible placement.

    The peak of the resultant BM always occurs at a load point: on each
    segment Mv and Mh are linear in x, so Mres^2 is a convex quadratic and
    is maximised at a segment end; the ends are the supports (M = 0) and
    the load points. So each candidate needs only two evaluations, which
    lets the whole sweep run as one vectorised numpy expression.
    """
    a_vals = np.arange(margin, L - margin - gap + 1e-9, step)
    b_vals = np.arange(margin + gap, L - margin + 1e-9, step)
    A, B = np.meshgrid(a_vals, b_vals, indexing="ij")
    feasible = (B - A) >= gap

    RAv = (fy2 * (L - A) + fy3 * (L - B)) / L
    RAh = (fx2 * (L - A) + fx3 * (L - B)) / L
    M_at_A = np.hypot(RAv * A, RAh * A)
    M_at_B = np.hypot(RAv * B - fy2 * (B - A), RAh * B - fx2 * (B - A))
    Mmax = np.maximum(M_at_A, M_at_B)
    Mmax[~feasible] = np.nan

    i, j = np.unravel_index(np.nanargmin(Mmax), Mmax.shape)
    return a_vals, b_vals, Mmax, float(A[i, j]), float(B[i, j]), float(Mmax[i, j])


def snap_up(value, options):
    for v in options:
        if v >= value:
            return v
    return None


# ---------------------------------------------------------------
# Page
# ---------------------------------------------------------------
st.title("Intermediate Shaft Design of a Two-Stage Spur Gearbox")
st.markdown(
    "**ATLAS** - Automated Torque and Load Analysis System. "
    "The intermediate shaft of a two-stage reduction gearbox carries the "
    "driven gear of stage 1 (gear 2) and the driving pinion of stage 2 "
    "(pinion 3) between two ball bearings A and B. Tooth loads at both "
    "meshes bend the shaft in two planes while the full torque is "
    "transmitted between the gears. The shaft is sized by the ASME code "
    "equation, checked for rigidity at the gear meshes and bearing seats, "
    "and the bearings are selected from the resulting reactions."
)

st.header("1. System layout and problem statement")
st.markdown(
    "- **Objective:** for a given power, speed and gear geometry, find the "
    "tooth loads, draw the SFD/BMD in both planes, size the shaft by "
    "strength and rigidity, and select bearings.\n"
    "- **Free body:** simply supported shaft (bearings at x = 0 and x = L), "
    "two transverse point loads (the resolved mesh forces at x = a and "
    "x = b), torque applied at gear 2 and removed at pinion 3.\n"
    "- **Method:** closed-form beam statics and superposition - no numerical "
    "integration anywhere, so every number is exactly reproducible by hand."
)

# ---------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## Design inputs")

    st.markdown("**1. Power flow**")
    P_kw = st.slider("Transmitted power (kW)", 1.0, 50.0, 10.0, step=0.5)
    N_in = st.slider("Input speed (rpm)", 500, 3000, 1440, step=10,
                     help="1440 rpm is the synchronous-slip speed of a "
                          "4-pole induction motor on a 50 Hz supply.")
    i1 = st.slider("Stage 1 ratio", 1.5, 6.0, 3.5, step=0.1,
                   help="Sets the intermediate shaft speed N2 = N_in / i1. "
                        "Stage 2 does not change the loads on this shaft.")

    st.markdown("**2. Gear geometry**")
    phi_deg = st.selectbox("Pressure angle (deg)", [14.5, 20.0, 25.0], index=1)
    m2 = st.select_slider("Gear 2 module (mm)", options=[2, 2.5, 3, 4, 5, 6, 8, 10], value=3)
    z2 = st.slider("Gear 2 teeth", 30, 120, 60)
    m3 = st.select_slider("Pinion 3 module (mm)", options=[2, 2.5, 3, 4, 5, 6, 8, 10], value=4)
    z3 = st.slider("Pinion 3 teeth", 17, 60, 24,
                   help="17 is the minimum to avoid interference at 20 deg "
                        "full depth.")
    theta2 = st.slider("Angular position of pinion 1 around shaft (deg)", 0, 345, 90, step=15,
                       help="Where the stage-1 pinion sits around this shaft, "
                            "CCW from horizontal. Sets the direction of the "
                            "mesh forces at gear 2.")
    theta3 = st.slider("Angular position of gear 4 around shaft (deg)", 0, 345, 270, step=15)
    rev2 = st.checkbox("Reverse Ft direction at gear 2", value=False,
                       help="The tangential force sense depends on rotation "
                            "direction; flip if your rotation is opposite.")
    rev3 = st.checkbox("Reverse Ft direction at pinion 3", value=True)

    st.markdown("**3. Shaft layout**")
    L = st.slider("Bearing span L (mm)", 250, 800, 400, step=10)
    a = st.slider("Gear 2 position a (mm)", 40, L - 80, min(100, L - 80), step=5)
    b = st.slider("Pinion 3 position b (mm)", a + 40, L - 40,
                  min(max(300, a + 40), L - 40), step=5)

    st.markdown("**4. Material and service factors**")
    mat_name = st.selectbox("Shaft material", list(MATERIALS))
    if MATERIALS[mat_name] is None:
        Sut = st.slider("Sut (MPa)", 400, 1200, 630, step=10)
        Syt = st.slider("Syt (MPa)", 250, 1000, 380, step=10)
    else:
        Sut, Syt = MATERIALS[mat_name]
    Kb = st.slider("Bending shock factor Kb", 1.0, 3.0, 1.5, step=0.05,
                   help="ASME: 1.5 for gradually applied load on a rotating "
                        "shaft, up to 2.0 for minor shock, 3.0 heavy shock.")
    Kt = st.slider("Torsion shock factor Kt", 1.0, 3.0, 1.0, step=0.05)
    keyway = st.checkbox("Keyways at gear seats (reduce tau by 25%)", value=True)
    L10h = st.slider("Required bearing life L10h (hours)", 4000, 40000, 10000, step=1000)

    # derived quantities
    N2 = N_in / i1
    T2 = 9.55e6 * P_kw / N2                      # N mm
    tau_allow = min(0.30 * Syt, 0.18 * Sut) * (0.75 if keyway else 1.0)
    d2g, d3g = m2 * z2, m3 * z3                  # pitch circle diameters

    st.markdown("---")
    st.caption("Derived quantities used in the design")
    st.markdown(f"Intermediate shaft speed N2 = **{N2:.0f} rpm**")
    st.markdown(f"Shaft torque T = **{T2/1e3:.1f} N m**")
    st.markdown(f"Pitch circles: gear 2 **{d2g:.0f} mm**, pinion 3 **{d3g:.0f} mm**")
    st.markdown(f"Allowable shear stress = **{tau_allow:.1f} MPa**")

# ---------------------------------------------------------------
# Forces
# ---------------------------------------------------------------
phi = np.radians(phi_deg)
Ft2, Fr2 = 2 * T2 / d2g, 2 * T2 / d2g * np.tan(phi)
Ft3, Fr3 = 2 * T2 / d3g, 2 * T2 / d3g * np.tan(phi)
fx2, fy2 = resolve_mesh_force(Ft2, Fr2, theta2, rev2)
fx3, fy3 = resolve_mesh_force(Ft3, Fr3, theta3, rev3)

loads_V = [(a, fy2), (b, fy3)]
loads_H = [(a, fx2), (b, fx3)]

xs = np.unique(np.concatenate([np.linspace(0.0, L, 801), [a, b]]))
Vv, Mv, RAv, RBv = beam_response(loads_V, L, xs)
Vh, Mh, RAh, RBh = beam_response(loads_H, L, xs)
Mres = np.hypot(Mv, Mh)
Mmax = float(np.max(Mres))
RA_res, RB_res = float(np.hypot(RAv, RAh)), float(np.hypot(RBv, RBh))

# ---------------------------------------------------------------
# Layout schematic and end-view force directions
# ---------------------------------------------------------------
sc1, sc2 = st.columns([3, 2])

with sc1:
    st.markdown("**Shaft layout (side view, true proportions)**")
    fig_lay = go.Figure()
    fig_lay.add_shape(type="rect", x0=0, x1=L, y0=-9, y1=9,
                      fillcolor="#c9ced4", line=dict(color="#555", width=1))
    for pos, dg, col, name in [(a, d2g, BLUE, "Gear 2"), (b, d3g, RED, "Pinion 3")]:
        fig_lay.add_shape(type="rect", x0=pos - 12, x1=pos + 12,
                          y0=-dg / 2, y1=dg / 2,
                          fillcolor=col, opacity=0.35, line=dict(color=col, width=2))
        fig_lay.add_annotation(x=pos, y=dg / 2 + 14, text=name,
                               showarrow=False, font=dict(color=col, size=12))
    fig_lay.add_trace(go.Scatter(
        x=[0, L], y=[-14, -14], mode="markers+text",
        marker=dict(symbol="triangle-up", size=15, color="#444"),
        text=["Bearing A", "Bearing B"], textposition="bottom center",
        textfont=dict(color=INK), showlegend=False))
    for x0, x1, lbl, yy in [(0, a, "a", -60), (0, b, "b", -80)]:
        fig_lay.add_annotation(x=x1, y=yy, ax=x0, ay=yy, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowcolor="#777")
        fig_lay.add_annotation(x=(x0 + x1) / 2, y=yy + 10, text=lbl,
                               showarrow=False, font=dict(color="#555"))
    fig_lay.update_layout(
        **PLOT_LAYOUT, height=330,
        xaxis=dict(title="x (mm)", **AXIS_STYLE, showgrid=False,
                   range=[-30, L + 30]),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        showlegend=False)
    st.plotly_chart(fig_lay, theme=None)

with sc2:
    st.markdown("**End view: mesh force directions**")
    fig_end = go.Figure()
    fig_end.add_shape(type="circle", x0=-20, x1=20, y0=-20, y1=20,
                      fillcolor="#c9ced4", line=dict(color="#555"))
    for th_deg, rev, col, tag in [(theta2, rev2, BLUE, "2"), (theta3, rev3, RED, "3")]:
        t = np.radians(th_deg)
        s = -1.0 if rev else 1.0
        px, py = 30 * np.cos(t), 30 * np.sin(t)
        # radial force arrow: along line of centres, toward the shaft
        fig_end.add_annotation(x=22 * np.cos(t), y=22 * np.sin(t),
                               ax=70 * np.cos(t), ay=70 * np.sin(t),
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowwidth=2,
                               arrowcolor=col)
        fig_end.add_annotation(x=84 * np.cos(t), y=84 * np.sin(t),
                               text=f"Fr{tag}", showarrow=False,
                               font=dict(color=col, size=13))
        # tangential force arrow: perpendicular to line of centres
        tx, ty = s * (-np.sin(t)), s * np.cos(t)
        fig_end.add_annotation(x=px + 45 * tx, y=py + 45 * ty,
                               ax=px, ay=py,
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowwidth=2,
                               arrowcolor=col)
        fig_end.add_annotation(x=px + 58 * tx, y=py + 58 * ty,
                               text=f"Ft{tag}", showarrow=False,
                               font=dict(color=col, size=13))
    fig_end.update_layout(
        **PLOT_LAYOUT, height=330,
        xaxis=dict(visible=False, range=[-105, 105]),
        yaxis=dict(visible=False, range=[-105, 105],
                   scaleanchor="x", scaleratio=1),
        showlegend=False)
    st.plotly_chart(fig_end, theme=None)
    st.caption("Radial force along the line of centres toward the shaft; "
               "tangential perpendicular to it. These are resolved into the "
               "vertical and horizontal load sets.")

# ---------------------------------------------------------------
# Gear forces table
# ---------------------------------------------------------------
st.header("2. Power flow and gear tooth loads")
st.markdown(
    f"The intermediate shaft turns at N2 = {N_in:.0f} / {i1:.1f} = "
    f"**{N2:.0f} rpm** and carries T = 9.55 x 10^6 P / N2 = "
    f"**{T2/1e3:.1f} N m**. Both meshes transmit this same torque, so the "
    f"smaller pinion sees the larger tangential force - which is why the "
    f"stage-2 mesh usually dominates the bending."
)
force_table = pd.DataFrame([
    {"Mesh": "Gear 2 (driven, stage 1)", "PCD (mm)": f"{d2g:.0f}",
     "Ft = 2T/d (N)": f"{Ft2:,.0f}", "Fr = Ft tan(phi) (N)": f"{Fr2:,.0f}",
     "Horizontal comp. (N)": f"{fx2:+,.0f}", "Vertical comp. (N)": f"{fy2:+,.0f}"},
    {"Mesh": "Pinion 3 (driving, stage 2)", "PCD (mm)": f"{d3g:.0f}",
     "Ft = 2T/d (N)": f"{Ft3:,.0f}", "Fr = Ft tan(phi) (N)": f"{Fr3:,.0f}",
     "Horizontal comp. (N)": f"{fx3:+,.0f}", "Vertical comp. (N)": f"{fy3:+,.0f}"},
])
st.table(force_table.set_index("Mesh"))

# ---------------------------------------------------------------
# SFD / BMD / TMD
# ---------------------------------------------------------------
st.header("3. Shear force, bending moment and torque diagrams")

pc1, pc2 = st.columns(2)
with pc1:
    fig_sfd = go.Figure()
    fig_sfd.add_hline(y=0, line=dict(color="#999", width=1))
    fig_sfd.add_trace(go.Scatter(x=xs, y=Vv, name="Vertical plane",
                                 line=dict(color=BLUE, width=2)))
    fig_sfd.add_trace(go.Scatter(x=xs, y=Vh, name="Horizontal plane",
                                 line=dict(color=RED, width=2)))
    fig_sfd.update_layout(**PLOT_LAYOUT, height=330,
                          title=dict(text="Shear force", font=dict(color=INK)),
                          xaxis=dict(title="x (mm)", **AXIS_STYLE),
                          yaxis=dict(title="V (N)", **AXIS_STYLE),
                          legend=dict(x=0.02, y=0.98, **LEGEND_BOX))
    st.plotly_chart(fig_sfd, theme=None)

with pc2:
    fig_bmd = go.Figure()
    fig_bmd.add_hline(y=0, line=dict(color="#999", width=1))
    fig_bmd.add_trace(go.Scatter(x=xs, y=Mv / 1e3, name="Mv",
                                 line=dict(color=BLUE, width=1.5, dash="dot")))
    fig_bmd.add_trace(go.Scatter(x=xs, y=Mh / 1e3, name="Mh",
                                 line=dict(color=RED, width=1.5, dash="dot")))
    fig_bmd.add_trace(go.Scatter(x=xs, y=Mres / 1e3,
                                 name="Resultant sqrt(Mv^2+Mh^2)",
                                 line=dict(color=INK, width=3)))
    fig_bmd.update_layout(**PLOT_LAYOUT, height=330,
                          title=dict(text="Bending moment", font=dict(color=INK)),
                          xaxis=dict(title="x (mm)", **AXIS_STYLE),
                          yaxis=dict(title="M (N m)", **AXIS_STYLE),
                          legend=dict(x=0.02, y=0.98, **LEGEND_BOX))
    st.plotly_chart(fig_bmd, theme=None)

fig_tmd = go.Figure()
fig_tmd.add_hline(y=0, line=dict(color="#999", width=1))
xt = [0, a, a, b, b, L]
yt = [0, 0, T2 / 1e3, T2 / 1e3, 0, 0]
fig_tmd.add_trace(go.Scatter(x=xt, y=yt, line=dict(color="#6d4c9f", width=2.5),
                             fill="tozeroy", fillcolor="rgba(109,76,159,0.12)",
                             showlegend=False))
fig_tmd.update_layout(**PLOT_LAYOUT, height=230,
                      title=dict(text="Torque (input at gear 2, output at pinion 3)",
                                 font=dict(color=INK)),
                      xaxis=dict(title="x (mm)", **AXIS_STYLE),
                      yaxis=dict(title="T (N m)", **AXIS_STYLE))
st.plotly_chart(fig_tmd, theme=None)

i_max = int(np.argmax(Mres))
st.markdown(
    f"Peak resultant bending moment **M = {Mmax/1e3:.1f} N m at "
    f"x = {xs[i_max]:.0f} mm** (it always falls at a gear location: between "
    f"loads, Mv and Mh are linear, so the resultant is convex on each "
    f"segment and peaks at a segment end). The critical section carries "
    f"this M together with the full torque T = {T2/1e3:.1f} N m."
)

# ---------------------------------------------------------------
# ASME diameter
# ---------------------------------------------------------------
st.header("4. Shaft diameter by the ASME code (strength)")
with st.expander("Design equation"):
    st.latex(r"\tau_{allow} = \min(0.30\,S_{yt},\ 0.18\,S_{ut}) \times 0.75 \text{ (if keyed)}")
    st.latex(r"d^3 = \frac{16}{\pi\,\tau_{allow}}\sqrt{(K_b M)^2 + (K_t T)^2}")
    st.markdown(
        "This is the maximum shear stress theory applied to combined steady "
        "bending and torsion, with the ASME shock factors standing in for "
        "load fluctuation. Note that although M itself is steady, a rotating "
        "shaft sees it as fully reversed bending at every fibre - which is "
        "why the code multiplies M by Kb and why a proper fatigue "
        "(Soderberg) check is listed as future work in section 9."
    )

d_strength = asme_diameter(Mmax, T2, Kb, Kt, tau_allow)

# ---------------------------------------------------------------
# Rigidity check and final selection
# ---------------------------------------------------------------
st.header("5. Rigidity check and final size selection")
st.markdown(
    "Gear shafts are usually governed by stiffness, not strength: excessive "
    "deflection at a mesh destroys the tooth contact pattern and excessive "
    "slope at a bearing seat causes edge loading. Limits used: deflection at "
    "each gear <= 0.01 x module of that gear, slope at each bearing <= "
    f"{SLOPE_LIMIT:.3f} rad (deep-groove ball bearings). Since deflection "
    "scales exactly with 1/d^4, one trial solve gives the required diameter "
    "in closed form: d_req = d_trial x (worst ratio)^(1/4)."
)

d_trial = max(d_strength, 20.0)
I_trial = np.pi * d_trial ** 4 / 64
yv, thAv, thBv = beam_deflection(loads_V, L, E_STEEL * I_trial, xs)
yh, thAh, thBh = beam_deflection(loads_H, L, E_STEEL * I_trial, xs)
defl_g2 = float(np.hypot(np.interp(a, xs, yv), np.interp(a, xs, yh)))
defl_g3 = float(np.hypot(np.interp(b, xs, yv), np.interp(b, xs, yh)))
slope_A = float(np.hypot(thAv, thAh))
slope_B = float(np.hypot(thBv, thBh))
worst_ratio = max(defl_g2 / (0.01 * m2), defl_g3 / (0.01 * m3),
                  slope_A / SLOPE_LIMIT, slope_B / SLOPE_LIMIT)
d_rigidity = d_trial * worst_ratio ** 0.25

d_needed = max(d_strength, d_rigidity)
d_sel = snap_up(d_needed, STD_DIAMETERS)
governing = "rigidity" if d_rigidity > d_strength else "strength"

if d_sel is None:
    st.error(
        f"Required diameter {d_needed:.0f} mm exceeds the preferred-size "
        f"list ({STD_DIAMETERS[-1]} mm). Shorten the bearing span, move the "
        f"gears toward the bearings (see section 7), or raise the modules."
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("d required (strength)", f"{d_strength:.1f} mm")
c2.metric("d required (rigidity)", f"{d_rigidity:.1f} mm")
c3.metric("Selected standard size", f"{d_sel:.0f} mm")
c4.metric("Governing criterion", governing)

# verify everything at the selected size
I_sel = np.pi * d_sel ** 4 / 64
scale = (d_trial / d_sel) ** 4          # deflections scale with 1/d^4
yv_s, yh_s = yv * scale, yh * scale
y_res = np.hypot(yv_s, yh_s)
defl_g2_s, defl_g3_s = defl_g2 * scale, defl_g3 * scale
slope_A_s, slope_B_s = slope_A * scale, slope_B * scale
sigma_b = 32 * Mmax / (np.pi * d_sel ** 3)
tau_t = 16 * T2 / (np.pi * d_sel ** 3)
tau_eq = 16 / (np.pi * d_sel ** 3) * np.hypot(Kb * Mmax, Kt * T2)

check_rows = [
    ("ASME equivalent shear stress", f"{tau_eq:.1f} MPa", f"{tau_allow:.1f} MPa", tau_eq <= tau_allow),
    ("Bending stress 32M/pi d^3", f"{sigma_b:.1f} MPa", "-", True),
    ("Torsional stress 16T/pi d^3", f"{tau_t:.1f} MPa", "-", True),
    ("Deflection at gear 2", f"{defl_g2_s:.4f} mm", f"{0.01 * m2:.3f} mm", defl_g2_s <= 0.01 * m2),
    ("Deflection at pinion 3", f"{defl_g3_s:.4f} mm", f"{0.01 * m3:.3f} mm", defl_g3_s <= 0.01 * m3),
    ("Slope at bearing A", f"{slope_A_s * 1e3:.3f} mrad", f"{SLOPE_LIMIT * 1e3:.1f} mrad", slope_A_s <= SLOPE_LIMIT),
    ("Slope at bearing B", f"{slope_B_s * 1e3:.3f} mrad", f"{SLOPE_LIMIT * 1e3:.1f} mrad", slope_B_s <= SLOPE_LIMIT),
]
check_df = pd.DataFrame(
    [{"Check": r[0], "Value": r[1], "Limit": r[2],
      "Status": "PASS" if r[3] else "FAIL"} for r in check_rows])
st.table(check_df.set_index("Check"))

all_pass = all(r[3] for r in check_rows)
if all_pass:
    st.success(
        f"**Selected shaft: {d_sel:.0f} mm, {mat_name.split('(')[0].strip()}** - "
        f"all strength and rigidity checks pass. The {governing} criterion "
        f"governed the size"
        + (", which is typical for gear shafts: stress alone would have "
           f"allowed {d_strength:.0f} mm." if governing == "rigidity" else ".")
    )
else:
    st.warning("A check fails at the selected standard size - this can "
               "happen when snapping lands exactly on a limit. Take the "
               "next standard size up.")

fig_defl = go.Figure()
fig_defl.add_hline(y=0, line=dict(color="#999", width=1))
fig_defl.add_trace(go.Scatter(x=xs, y=y_res, name="Resultant deflection",
                              line=dict(color=INK, width=2.5)))
fig_defl.add_trace(go.Scatter(
    x=[a, b], y=[defl_g2_s, defl_g3_s], mode="markers+text",
    text=[f"gear 2: {defl_g2_s:.3f}", f"pinion 3: {defl_g3_s:.3f}"],
    textposition="top center", textfont=dict(color=INK),
    marker=dict(size=10, color=RED), name="At gear meshes"))
fig_defl.update_layout(**PLOT_LAYOUT, height=300,
                       title=dict(text=f"Resultant deflection at d = {d_sel:.0f} mm",
                                  font=dict(color=INK)),
                       xaxis=dict(title="x (mm)", **AXIS_STYLE),
                       yaxis=dict(title="deflection (mm)", **AXIS_STYLE),
                       legend=dict(x=0.02, y=0.98, **LEGEND_BOX))
st.plotly_chart(fig_defl, theme=None)

st.markdown("---")

# ---------------------------------------------------------------
# Bearings
# ---------------------------------------------------------------
st.header("6. Bearing selection")
st.markdown(
    "Spur gears produce no axial thrust, so the bearings carry pure radial "
    "load equal to the resultant reactions. For ball bearings the "
    "life-load relation is L10 = (C/P)^3 million revolutions, so the "
    "required dynamic capacity is C = P x (60 N2 L10h / 10^6)^(1/3)."
)
life_mrev = 60 * N2 * L10h / 1e6
R_worst = max(RA_res, RB_res)
C_req = R_worst * life_mrev ** (1 / 3) / 1000.0   # kN

bore = snap_up(d_sel, sorted(BEARING_TABLE))
bc1, bc2, bc3, bc4 = st.columns(4)
bc1.metric("Reaction at A", f"{RA_res:,.0f} N")
bc2.metric("Reaction at B", f"{RB_res:,.0f} N")
bc3.metric("Required C (worse seat)", f"{C_req:.1f} kN")
bc4.metric("Bearing bore", f"{bore} mm" if bore else "beyond table")

if bore is None:
    st.error("Shaft is larger than the bearing table covers - consult a "
             "full catalogue for 65+ mm bores.")
else:
    candidates = BEARING_TABLE[bore]
    pick = next(((n, C) for n, C in candidates if C >= C_req), None)
    rows = [{"Designation": n, "Bore (mm)": bore, "C (kN)": C,
             "Life at this load (h)": f"{(C * 1000 / R_worst) ** 3 * 1e6 / (60 * N2):,.0f}",
             "Adequate": "yes" if C >= C_req else "no"} for n, C in candidates]
    st.table(pd.DataFrame(rows).set_index("Designation"))
    if pick:
        st.success(f"**Select {pick[0]}** (C = {pick[1]:.1f} kN >= "
                   f"{C_req:.1f} kN required) at both seats; using the "
                   f"worse reaction at both keeps the parts identical.")
    else:
        st.warning("Neither series meets the required capacity at this bore - "
                   "reduce the required life, or use a roller bearing.")

st.markdown("---")

# ---------------------------------------------------------------
# Placement optimisation
# ---------------------------------------------------------------
st.header("7. Gear placement optimisation")
st.markdown(
    "With the tooth loads fixed by the gear geometry, the only free layout "
    "variables are the gear positions a and b. Every feasible placement "
    "(40 mm edge margins, 40 mm minimum gap for face widths) is evaluated "
    "below. Because the peak resultant moment occurs at a load point (the "
    "convexity argument in section 3), each candidate needs just two "
    "evaluations, so the full sweep is a single vectorised expression."
)
a_vals, b_vals, Msweep, a_opt, b_opt, M_opt = placement_sweep(
    L, fx2, fy2, fx3, fy3, margin=40.0, gap=40.0)

d_str_opt = asme_diameter(M_opt, T2, Kb, Kt, tau_allow)
reduction = (1 - M_opt / Mmax) * 100

fig_opt = go.Figure(data=go.Heatmap(
    z=Msweep.T / 1e3, x=a_vals, y=b_vals, colorscale="Viridis",
    colorbar=dict(title="Mmax (N m)", tickfont=dict(color=INK))))
fig_opt.add_trace(go.Scatter(x=[a], y=[b], mode="markers+text",
                             marker=dict(symbol="x", size=13, color="white",
                                         line=dict(width=2)),
                             text=["current"], textposition="top center",
                             textfont=dict(color="white"), showlegend=False))
fig_opt.add_trace(go.Scatter(x=[a_opt], y=[b_opt], mode="markers+text",
                             marker=dict(symbol="star", size=16, color=RED),
                             text=["optimum"], textposition="bottom center",
                             textfont=dict(color=RED), showlegend=False))
fig_opt.update_layout(**PLOT_LAYOUT, height=430,
                      xaxis=dict(title="Gear 2 position a (mm)", **AXIS_STYLE),
                      yaxis=dict(title="Pinion 3 position b (mm)", **AXIS_STYLE))
st.plotly_chart(fig_opt, theme=None)

st.markdown(
    f"- Current placement (a = {a}, b = {b}): Mmax = **{Mmax/1e3:.1f} N m**, "
    f"strength diameter {d_strength:.1f} mm.\n"
    f"- Optimum (a = {a_opt:.0f}, b = {b_opt:.0f}): Mmax = "
    f"**{M_opt/1e3:.1f} N m** ({reduction:.0f}% lower), strength diameter "
    f"{d_str_opt:.1f} mm.\n"
    f"- The optimum pushes both gears toward the bearings, which is exactly "
    f"what beam intuition predicts: bending moment under a point load "
    f"scales with the product of its distances to the two supports. "
    f"Deflection at the meshes improves for the same reason, so the "
    f"rigidity-governed size shrinks too. In a real gearbox, casing and "
    f"assembly clearances set how far this can be pushed."
)

st.markdown("---")

# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------
st.header("8. Validation against textbook formulas")
st.markdown(
    "The same beam routines used everywhere above are checked against the "
    "closed-form results for a single central load P on the current span: "
    "Mmax = PL/4, ymax = PL^3/48EI, end slope = PL^2/16EI. Errors should "
    "sit at machine precision; anything larger means the statics or the "
    "superposition formulas are wrong."
)
P_val = 1000.0
xv = np.unique(np.concatenate([np.linspace(0.0, L, 2001), [L / 2]]))
_, M_val, RA_val, RB_val = beam_response([(L / 2, P_val)], L, xv)
y_val, th_val, _ = beam_deflection([(L / 2, P_val)], L, E_STEEL * I_sel, xv)
err_M = abs(float(np.max(M_val)) - P_val * L / 4)
err_y = abs(float(np.max(y_val)) - P_val * L ** 3 / (48 * E_STEEL * I_sel))
err_th = abs(th_val - P_val * L ** 2 / (16 * E_STEEL * I_sel))
eq_res = abs(RAv + RBv - (fy2 + fy3)) + abs(RAh + RBh - (fx2 + fx3))

v1, v2, v3, v4 = st.columns(4)
v1.metric("|Mmax - PL/4|", f"{err_M:.2e} N mm")
v2.metric("|ymax - PL^3/48EI|", f"{err_y:.2e} mm")
v3.metric("|slope - PL^2/16EI|", f"{err_th:.2e} rad")
v4.metric("Equilibrium residual (actual case)", f"{eq_res:.2e} N")

st.markdown("---")

# ---------------------------------------------------------------
# Assumptions and limitations
# ---------------------------------------------------------------
st.header("9. Assumptions and limitations")
with st.expander("Full list", expanded=False):
    st.markdown(f"""
**Modelling assumptions:**
- Bearings act as simple supports: they carry force but no moment, and the
  shaft is free to rotate over them. This is the standard idealisation for
  deep-groove ball bearings.
- The shaft is a uniform circular section of the selected diameter over the
  full span. A production shaft is stepped; the steps raise stiffness near
  the middle and introduce stress concentration at shoulders - a fillet-by-
  fillet check with Kt factors is the natural next step.
- Spur gears only, so no axial thrust on the bearings. Helical gears would
  add a thrust component and a moment from the offset radial force.
- Gear and shaft self-weight neglected (a few kg against multi-kN tooth
  loads).
- The tangential force sense depends on rotation direction; the reverse
  toggles in the sidebar cover both cases rather than hard-coding one.

**Design-code notes:**
- The ASME shock factors Kb and Kt approximate load fluctuation. A rotating
  shaft sees the steady bending moment as fully reversed stress, so a
  Soderberg or Goodman fatigue check with surface, size and reliability
  factors is the rigorous treatment - stated here as future work.
- Rigidity limits used: 0.01 x module transverse deflection at a spur mesh
  and {SLOPE_LIMIT:.3f} rad slope at a deep-groove ball bearing. These are
  widely used guidelines, not statutory limits.
- Bearing capacities are representative catalogue values for 62/63 series;
  a real selection would confirm against the manufacturer's current
  catalogue and check limiting speed and lubrication.
- Gear tooth strength (Lewis bending, wear) is not checked here - the
  modules are inputs. Tooth design is a separate calculation that would
  normally precede this shaft design.

**Why the peak resultant moment is found only at gear locations:** between
loads, Mv and Mh are both linear in x, so Mres^2 = Mv^2 + Mh^2 is a convex
quadratic on each segment and takes its maximum at a segment end. The ends
are the supports (where M = 0) and the load points - so checking the two
gear locations is exact, not an approximation. This is also what makes the
placement sweep in section 7 fast.
""")
