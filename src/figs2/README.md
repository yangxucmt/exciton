# Figure S2 Data and Plotting Code

This folder contains the code and data used to generate Fig. S2.

## Fig. S2 — A1s exciton shift vector vs strain

File: "figs2.csv"

- "strainlist": Strain (%)
- "rshift_wannier": Computed using Eq.10 with Wannier and envelope functions only (polarization-independent).  
- "rshift_full_2pi_3": Computed using Eq.7 with light-polarization angle $\theta = 2\pi/3$.
- "rshift_full_3pi_4": Computed using Eq.7 with light-polarization angle $\theta = 3\pi/4$.
- "rshift_full_5pi_6": Computed using Eq.7 with light-polarization angle $\theta = 5\pi/6$.

All y-axis values correspond to the $y$-component of the A1s excitonic transition shift vector $R_y$ composed of spin-up bound particle-hole pairs.

Units: 
- x-axis: Strain in terms of percentage
- y-axis: Shift vector in units of $10^{-2}a$, where $a = 3.16\text{\AA}$.