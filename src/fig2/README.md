# Figure 2 Data and Plotting Code

This folder contains the code and data used to generate Fig. 2.

## Overview

The following states are analyzed across different system sizes and values of $\kappa$:

  - localized_1: exciton state with the lowest energy  
  - localized_2: exciton state with the 3rd lowest energy  
  - delocalized_1: delocalized state with the 10th lowest energy above the band gap  
  - delocalized_2: delocalized state with the 18th lowest energy above the band gap  
  - delocalized_3: delocalized state with the 35th lowest energy above the band gap  

Energies are measured relative to $\Delta$, so the band gap is located at energy 1.

---

## Fig. 2a — Energy versus flux $\kappa$

Data file: "fig2a.csv"

- "Ordinal": ordinal number of the energy level  
- "0.0", "0.1", …: energy levels (low to high) at $\kappa L/2\pi = 0.0$, $0.1$, … in units of $\Delta$ (band gap)  

Energy spectra are computed for a honeycomb lattice with a staggered potential and screened Coulomb interactions along the $x$ direction.  

At system size $\sqrt{N} = 24$, there are 20 states below the band gap. Therefore, delocalized_1, delocalized_2, and delocalized_3 correspond to the 30th, 38th, and 55th states, respectively.  

### Inset (angle dependence)

Data file: "fig2a_inset.csv"

- "size_list": linear system sizes $\sqrt{N}$, where $N$ is the total number of unit cells  
- "localized_1", "localized_2", "delocalized_1", "delocalized_2", "delocalized_3": states as defined above  

Values represent the Thouless number, defined as the energy difference of the corresponding state between $\kappa L/2\pi = 0$ and $\kappa L/2\pi = \pi$, divided by the typical level spacing $\Delta/N$, for the given system sizes.

Units:  
- x-axis: unitless (system size)  
- y-axis: unitless (Thouless number)

---

## Fig. 2b — Wilson phase versus flux $\kappa$

Data file: "fig2b.csv"

- "kappa_list": flux $\kappa L/2\pi$  
- "localized_1", "localized_2", "delocalized_1", "delocalized_2", "delocalized_3": states as defined above  

Values are the Wilson phase (phase of the Wilson loop) for the corresponding states.

Units:  
- x-axis: unitless ($\kappa L/2\pi$)
- y-axis: radians  

---

## Fig. 2c — Standard deviation of Wilson phases versus system size $\sqrt{N}$

Data file: "fig2c.csv"

- "size_list": linear system sizes $\sqrt{N}$, where $N$ is the total number of unit cells  
- "localized_1", "localized_2", "delocalized_1", "delocalized_2", "delocalized_3": states as defined above  

Values represent the standard deviation of the Wilson phase sampled at different $\kappa L/2\pi$ for each system size.

Units:  
- x-axis: unitless (system size)  
- y-axis: radians  