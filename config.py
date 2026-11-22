"""Global configuration for the occupancy gaussian framework."""

import numpy as np

# Langevin parameters
DT = 1e-4                 # timestep (s)
T = 300.0                 # temperature (K); set 0.0 for deterministic
KB = 1.380649e-23         # Boltzmann constant
DAMPING = 0.1             # friction coefficient gamma

# Material defaults (assign per-object in scenes)
RHO = 1200.0              # density kg/m^3
E_YOUNG = 1e6             # Pa
NU = 0.3                  # Poisson ratio

# Contact
PEN_THRESHOLD = 0.05      # max penetration as fraction of gaussian scale
FRICTION_MU = 0.3

# Modal analysis
N_MODES = 6               # lowest modes to compute

# Validation
ENERGY_DRIFT_LIMIT = 1e-3  # per second
RNG = np.random.default_rng(42)
