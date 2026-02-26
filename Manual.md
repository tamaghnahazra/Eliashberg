# Eliashberg Solver Manual

This manual catalogs the core codebase: `LinearizedEliashberg.py` and `EliashbergSr2RuO4.py`.

---

## 1. `LinearizedEliashberg.py`

### 1.1 Top-Level Functions

#### CORE
* `project_delta(psi, target_irrep, character_table, apply_symmetry=None)` --- Projects a wavefunction into a selected Irrep; use `target_irrep` (e.g., 'B1g') to filter symmetry sectors.

#### HELPER
* `character(kFF, nk)` --- Computes the symmetry character of `kFF` (rank-2 momentum tensor of form factor) on an `nk`x`nk` grid.
* `character_weighted(kFF, nk)` --- Returns $kFF_1 * kFF_2 / ((|kFF_1| + |kFF_2|)/2)^2$ to suppress numerical noise at nodes.
* `get_all_characters(Gamma, nk)` --- Extracts symmetry characters for a complete spin-orbital component structure.
* `get_max_abs_values(Gamma)` --- Returns the maximum absolute values among components to find dominant intensities.
* `get_SigmaLScomponents_1orb(matrix, isDelta=False)` --- Extracts singlet and triplet components from the first 2x2 spin block; `isDelta=True` applies pairing anomalous Green's function rules.
* `get_Gamma_1orb(LScomponents)` --- Groups the isolated 1-orbital Spin Orbital (LS) components.
* `prep_for_plot_SigmaRe_1orb(sigma_wk, nk, norb)` --- Formats Re $\Sigma$ by averaging the first positive and negative Matsubara frequency lobes.
* `prep_for_plot_Delta_1orb(vs, nk, norb, oddfreq=False, n_w=None)` --- Normalizes and slices gaps; `oddfreq=True` extracts the purely odd-frequency component. If `vs` is 1D, `n_w` must be provided for reshaping.
* `plot_Gamma_1orb(Gamma, nk, uniform_colorbar=False, round_character=True)` --- Renders spatial plots of components; `uniform_colorbar=True` forces consistent scales across subplots, `round_character=True` rounds characters to the nearest integer.

---

### 1.2 Classes

#### `EliashbergSolver`
General backbone for N-orbital interacting solvers with FFT-optimized self-consistent Dyson and gap equations.

#### CORE
* `dyson_solver(seed_sigma=None, zero_Gtau0=True)` --- Solves the Dyson loop self-consistently for $\Sigma$; `seed_sigma` provides a warm start, `zero_Gtau0=True` constrains $G(\tau=0)=0$ to remove Hartree shifts at all $i\omega_n$.
* `solve_linearized_gap_static(add_linewidth=False, solncount=1, v0=None, tol=1e-8, project_to=None)` --- Finds gap eigenvalues; `add_linewidth=True` adds a quasiparticle lifetime by using dynamical $G(k, \omega)$, `v0` is an optional initial guess, and `project_to` restricts the solver to a specific Irrep.
* `solve_linearized_gap_dynamic(solncount=1, v0=None, tol=1e-8, init_with_static=False, project_to=None)` --- Power iteration for the dynamical gap; `init_with_static=True` uses the static result to accelerate convergence.
* `nonlinear_dynamic_gap_solver(seed_delta=None, seed_sigma=None, iterations=500, zero_Gtau0=True, tol=1e-5, project_to=None)` --- Coupled gap and self-energy Nambu solver; `seed_delta`/`seed_sigma` provide warm starts (else initialized with noise), `iterations` sets the max loop count, and `project_to` enforces symmetry at each step.
* `calculate_hk(k1, k2, k3)` --- Abstract placeholder for subclass to define the model-specific tight-binding Hamiltonian.
* `calculate_gamma(wmesh_boson_kmesh, target_shape)` --- Abstract placeholder for subclass to define the $q,\Omega$-dependent pairing interaction.

#### HELPER
* `get_filling(gf_wk)` --- Computes total particle density from a Matsubara Green's function (normal sector only, not Nambu).
* `fourier_wk_to_tr(gf_wk)` / `fourier_tr_to_wk(gf_tr)` --- Performs optimized FFTs between $(\omega, k)$ and $(\tau, r)$.
* `fermion_antisymmetrize(Delta)` --- Enforces the required exchange symmetry $\Delta_{ab}(k, \omega) = -\Delta_{ba}(-k, -\omega)$.
* `apply_symmetry(op, Delta_k)` --- Transforms the gap matrix using model-specific symmetry unitaries (e.g., `op='sigma_d'`).
* `check_symmetries(eigvec, threshold, verbose)` --- Scans all point-group operations and prints characters, compares with the character table to identify the Irrep.
* `twoparticle_GG(h_k, g_wk, add_linewidth=True)` --- Computes the pairing bubble; `add_linewidth=True` adds a quasiparticle lifetime and uses numerical frequency sums, otherwise sums statically over $H_k$.

#### `SquareLattice` (Derived from `EliashbergSolver`)
Single-band implementation on a square lattice with spin-fluctuation interaction.

#### CORE
* `calculate_hk(k1, k2, k3)` --- Returns the nearest-neighbor tight-binding dispersion for a square cell.
* `calculate_gamma(...)` --- Computes the paramagnon susceptibility $\chi(q, \Omega)$ coupled via Pauli matrices.

#### HELPER
* `plot_quasiparticle_weight(figsize)` --- Visualizes the renormalization $Z = (1 - Im \Sigma/\omega)^{-1}$ across the zone.
* `plot_fermi_surface(figsize, nkmesh)` --- Visualizes $G(\omega_0) G(- \omega_0)$; use `nkmesh` to interpolate linearly onto a finer grid.
* `plot_delta(oddfreq=False)` --- Slices and plots the gap into 1-orbital components; automatically handles both nonlinear `delta_wk` and linearized `vs_dynamic` data.
* `plot_deltaMF(vs=None, idx=0)` --- Plots the static mean-field gap $\Delta(k)$ using internal `vs_static` or external `vs`.

---

## 2. `EliashbergSr2RuO4.py`

Multi-orbital implementation for the $\{yz, xz, xy\}$ band structure of $Sr_2RuO_4$.

### 2.1 Top-Level Functions

#### CORE
* `decompose_form_factor(kFF, nk)` --- Partitions a form factor into $s, d, p,$ and $g$ lattice symmetries using parity masking.

#### HELPER
* `xi(T)` --- Returns the temperature-dependent magnetic correlation length $\xi(T)$ for SRO.
* `get_SigmaLScomponents(matrix, isDelta=False)` --- Decomposes $6 \times 6$ tensors into Spin Orbital representations.
* `get_Gamma(LScomponents)` --- Groups Spin Orbital components into local Irreps (e.g., $LzSz \to A_1$), returning four lists for A1g, A2g, B1g, B2g.
* `prep_for_plot_Delta(vs, nw, nk, norb, oddfreq=False)` --- Formats multi-orbital eigenvectors for point-group analysis; `oddfreq` extracts the odd-frequency sector.
* `plot_Gamma(Gamma, nk, uniform_colorbar=False, round_character=True)` --- Visualizes dominant components and identifies the closest Irrep.

### 2.2 Classes

#### `SBSSr2RuO4` (Derived from `EliashbergSolver`)
3-orbital model including linear strain effects and multi-band paramagnon coupling.

#### CORE
* `tight_binding_parameters_strain(eps_xx, nu_xy, alpha, beta)` --- Returns hopping values adjusted for uniaxial strain $\epsilon_{xx}$.
* `calculate_hk(kx, ky, kz)` --- Constructs the 6x6 Hamiltonian matrix including $V_k$ hybridization.

#### HELPER
* `plot_quasiparticle_weight(orbital_resolved=False, figsize=(8, 7))` --- Displays the $Z$ factor; `orbital_resolved=True` separates plots by orbital channel.
* `plot_fermi_surface(figsize, nkmesh)` --- Interpolates and visualizes the multi-orbital Fermi surface.
* `plot_delta(oddfreq=False)` --- Plots the 6-orbital gap function; handles both `delta_wk` and `vs_dynamic`.
* `plot_deltaMF(vs=None, idx=0)` --- Plots the static multi-orbital gap.
* `plot_bandstructure(figsize=(8, 6))` --- Plots the dispersion along the high-symmetry path $\Gamma-X-M-\Gamma$.

#### `PSSr2RuO4` (Derived from `SBSSr2RuO4`)
Extended tight-binding parameterization from Palle & Schmalian.

#### CORE
* `calculate_hk(k1, k2, k3)` --- Implements the generalized dispersion with higher-order neighbors and SOC ($t_1$ to $t_{11}$ parameters).
