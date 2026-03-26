# Figure 3 Data and Plotting Code

This folder contains the code and data used to generate Fig. 3.

## Overview

- Fig. 3a,b: Schematic illustrations (no associated datasets).

---

## Fig. 3c — Free particle–hole shift vector vs strain

Data file: "fig3c.csv"

- "strain": Strain (in %)
- "rshift_theta_0": Shift vector at polarization angle $\theta = 0$
- "rshift_theta_pi_6": Shift vector at $\theta = \pi/6$
- "rshift_theta_pi_2": Shift vector at $\theta = \pi/2$

All values correspond to the $y$-component of the free particle–hole shift vector at $K$ point, for transitions between the valence band and lower conduction band (spin-up electrons).

Units: $10^{-2} a$, where $a = 3.16\text{\AA}$

### Inset (angle dependence)

Data file: "fig3c_inset.csv"

- "theta_list": Polarization angle (radians)
- "rshift_strain_0": Zero strain
- "rshift_strain_1percent": 1% strain

---

## Fig. 3d — A1s exciton shift vector vs strain

Data files:
- "fig3d_full.csv"
- "fig3d_wannier.csv"

All y-axis values correspond to the $y$-component of the A1s excitonic transition shift vector $R_y$ composed of spin-up bound particle-hole pairs.

Common units:
- x-axis: Strain (%)
- y-axis: Shift vector in units of $10^{-2}a$, where $a = 3.16\text{\AA}$. 


### Full calculation (Eq. 7)

File: "fig3d_full.csv"

- "strain"
- "rshift_full_2pi_3": $\theta = 2\pi/3$
- "rshift_full_3pi_4": $\theta = 3\pi/4$

Includes explicit light-polarization dependence.

### Wannier-based calculation (Eq. 10)

File: "fig3d_wannier.csv"

- "strain"
- "rshift_wannier"

Computed using Wannier and envelope functions only (polarization-independent).  
Evaluated on a dense strain grid $-3.0:0.06:3.0$ for the solid line.

### Inset (angle dependence)

Data file: "fig3d_inset.csv"

- "theta_list": Polarization angle (radians)
- "rshift_strain_0": Zero strain
- "rshift_strain_2_dot_66_percent": 2.66% strain (along $y$)

Computed using Eq. (7).

---

## Fig. 3e — Shift current (free particle–hole pairs)

Data file: "fig3e.csv"

- "th_angle_list": Polarization angle (radians)
- "shift_current_strain_0": Zero strain
- "shift_current_strain_2_dot_66_percent": 2.66% strain (along $y$)

Values correspond to the total shift current density along $y$ direction. In the calculation we sum over all bands.

Units: $nA/\mu m$.


---

## Fig. 3f — Shift current (A1s excitons)

Data file: "fig3f.csv"

- "th_angle_list": Polarization angle (radians)
- "shift_current_strain_0": Zero strain
- "shift_current_strain_2_dot_66_percent": 2.66% strain (along $y$)

Values correspond to the total shift current density along $y$ direction. In the calculation we sum over both spins.

Units: $nA/\mu m$.


---

## Notes

- All angles are in radians.
- All shift vectors are in units of $10^{-2} a$, with lattice constant $a = 3.16\text{\AA}$.
- All shift current density are in units of $nA/\mu m$.