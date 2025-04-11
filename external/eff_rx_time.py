#!/usr/bin/env python
"""
JMAK Model Helper Functions
----------------------------

This module implements methods for solving the inverse non‐isothermal JMAK problem.
In the JMAK kinetics model, the isothermal transformation is given by

    X(t, T) = 1 - exp{ - b(T)^n (t - t_inc(T))^n }

with:
    t_inc(T) = exp(a2 + B2/T)
    b(T)     = exp(a1 + B1/T)
    ln_term  = (-ln(1 - X*))^(1/n)

and the non‐isothermal (recrystallization) problem is addressed by splitting the 
operating time, τ_op, into an incubation period and a recrystallization period. 
Disturbance windows characterized by a transient temperature profile (ΔT_r(t)) 
occur over a total of N_d cycles, each of duration τ_r, over which the temperature 
is given by T_r(t; T_a) = T_a + ΔT_r(t). The solution searches for an "adjusted" 
isothermal temperature T_a so that the following residuals vanish:

  F1 = ((τ_a_inc - N_d,inc τ_r)/t_inc(T_a)) + N_d,inc Γ_inc^r(T_a) - 1   = 0
  F2 = ((τ_a_rx - N_d,rx τ_r) * b(T_a)) + N_d,rx Γ_rx^r(T_a) - S*         = 0

with S* = (-ln(1 - X*))^(1/n). The formulation uses numerical integration 
(via the trapezoidal rule) to approximate the disturbance contributions.

Usage:
    - Define a JMAKParams instance with the appropriate kinetic, scheduling, 
      and disturbance parameters.
    - Use `invert_isothermal_T` to compute the isothermal temperature that would 
      yield the target transformation at τ_op.
    - Use `solve_Ta_tau` to solve the 2D residual problem for a given number of 
      incubation cycles.
    - `search_solution_range` and `best_solution_soft` explore candidate integer 
      subdivisions (cycles) to determine the best transient solution.

Author: Michael Lanahan
Date: 2025-04-08
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Generator
import numpy as np
from scipy.optimize import root, bisect
from scipy.integrate import trapezoid


# --------------------------------------------------------------------
# 1. Parameter Container for JMAK Model
# --------------------------------------------------------------------
@dataclass
class JMAKParams:
    """
    Container for the JMAK kinetics parameters and process settings.
    
    Attributes:
        n (float): Exponent in the JMAK kinetics.
        a1 (float): Logarithmic pre-exponential factor for growth.
        B1 (float): Activation parameter for growth.
        a2 (float): Logarithmic pre-exponential factor for incubation.
        B2 (float): Activation parameter for incubation.
        X_star (float): Target transformed fraction.
        tau_op (float): Total operating time (seconds).
        tau_r (float): Duration of a disturbance (transient) window.
        N_d (int): Total number of disturbance windows.
        alpha (float): Cycle‑ratio parameter (nuisance parameter to balance incubation and recrystallization).
        deltaT_r (Optional[np.ndarray]): Array (shape: (Nt,)) of temperature deviations over a transient window.
        t (Optional[np.ndarray]): Time array corresponding to deltaT_r values.
        Tmin (float): Lower bound for temperature search (default: 600.0).
        Tmax (float): Upper bound for temperature search (default: 4000.0).
    """
    n: float
    a1: float
    B1: float         # growth related constant
    a2: float
    B2: float         # incubation related constant

    X_star: float
    tau_op: float     # operating time [s]

    tau_r: float      # disturbance (transient window) length [s]
    N_d: int          # total number of windows
    alpha: float      # cycle‑ratio parameter
    deltaT_r: Optional[np.ndarray] = None  # temperature deviations (transient profile)
    t: Optional[np.ndarray] = None         # time grid (assumed uniform for integration)

    Tmin: float = 600.0   # lower bound for T search (optional)
    Tmax: float = 4000.0  # upper bound for T search (optional)

    def t_inc(self, T: float) -> float:
        """
        Compute the incubation time given temperature T.
        
        Args:
            T (float): Temperature [K] (or appropriate unit).

        Returns:
            float: Incubation time computed as exp(a2 + B2/T).
        """
        return np.exp(self.a2 + self.B2 / T)

    def b(self, T: float) -> float:
        """
        Compute the growth factor b given temperature T.
        
        Args:
            T (float): Temperature [K].

        Returns:
            float: Growth factor computed as exp(a1 + B1/T).
        """
        return np.exp(self.a1 + self.B1 / T)

    @property
    def ln_term(self) -> float:
        """
        Precompute the logarithmic term:
        
            (-ln(1 - X_star))^(1/n)
            
        Returns:
            float: The computed logarithmic term.
        """
        return (-np.log1p(-self.X_star)) ** (1.0 / self.n)


# --------------------------------------------------------------------
# 2. Single-Window Integrals
# --------------------------------------------------------------------
def single_window_integrals(p: JMAKParams) -> Tuple:
    """
    Computes the integrals over a single disturbance window (if provided)
    for both incubation and recrystallization phases.

    The integrals computed are:
        I_inc_r(T) = ∫ (1/t_inc(T + ΔT_r)) dt
        I_rx_r(T)  = ∫ b(T + ΔT_r) dt

    Integration is performed using the trapezoidal rule over the provided time grid.

    Args:
        p (JMAKParams): The JMAK parameter container.

    Returns:
        Tuple[Callable, Callable]: A pair of functions that take temperature T
        and return the integrated values.
    """
    # If there is no transient profile, return zero integrals.
    if p.tau_r == 0.0 or p.N_d == 0 or p.deltaT_r is None:
        return (lambda T: 0.0), (lambda T: 0.0)

    # Assume uniform time grid from provided time array.
    dt = np.diff(p.t)[0]

    def I_inc_r(T: float) -> float:
        """Computes the incubation integral for a disturbance window."""
        T_prof = T + p.deltaT_r  # add the transient temperature change
        return trapezoid(1.0 / p.t_inc(T_prof), dx=dt)

    def I_rx_r(T: float) -> float:
        """Computes the recrystallization integral for a disturbance window."""
        T_prof = T + p.deltaT_r
        return trapezoid(p.b(T_prof), dx=dt)

    return I_inc_r, I_rx_r


# --------------------------------------------------------------------
# 3. Inverted Isothermal JMAK Function
# --------------------------------------------------------------------
def invert_isothermal_T(
    params: JMAKParams,
    T_low: float = 200.0,
    T_high: float = 3000.0,
    tol: float = 1e-8,
) -> float:
    """
    Finds the isothermal temperature T_iso such that
    t_iso(X = X_star, T = T_iso) = tau_op
    
    The isothermal time is defined as:
        t_iso = t_inc(T) + (1/b(T)) * (-ln(1 - X_star))^(1/n)

    Args:
        params (JMAKParams): The model parameters.
        T_low (float): Lower bound of temperature for bisection.
        T_high (float): Upper bound of temperature for bisection.
        tol (float): Tolerance for the bisection method.

    Returns:
        float: The isothermal temperature that meets the target operating time.
    """
    ln_term = params.ln_term

    def g(T: float) -> float:
        return params.t_inc(T) + ln_term / params.b(T) - params.tau_op

    # Use bisection to invert the isothermal function.
    return bisect(g, T_low, T_high, xtol=tol)


# --------------------------------------------------------------------
# 4. 2D Root Solver for Fixed Integer k
# --------------------------------------------------------------------
def solve_Ta_tau(
    k: int,
    p: JMAKParams,
    I_inc_r,
    I_rx_r,
    tol: float = 1e-10
) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Solves for the adjusted isothermal temperature T_a and the incubation time 
    tau_inc for a fixed number of incubation cycles (k), ensuring that the residuals
    (F1, F2) in the non-isothermal formulation vanish.

    The residuals are:
        F1 = ((tau_inc - k*tau_r)/t_inc(T)) + k * I_inc_r(T) - 1
        F2 = ((tau_rx - (N_d - k) * tau_r) * b(T)) + (N_d - k) * I_rx_r(T) - ln_term
        
    where:
        tau_rx = tau_op - tau_inc
        ln_term = (-ln(1 - X_star))^(1/n)

    Args:
        k (int): Number of incubation cycles (disturbance windows used in incubation).
        p (JMAKParams): Model parameters.
        I_inc_r: Function to compute the incubation integral for a disturbance window.
        I_rx_r: Function to compute the recrystallization integral for a disturbance window.
        tol (float): Tolerance for the root finding method.

    Returns:
        Tuple containing:
            - success (bool): True if the solution converged.
            - T (Optional[float]): Adjusted isothermal temperature T_a.
            - tau_inc (Optional[float]): Computed incubation time.
            - tau_rx (Optional[float]): Recrystallization time (tau_op - tau_inc).
            - F2_abs (Optional[float]): Absolute residual of F2.
    """
    if k * p.tau_r >= p.tau_op:
        # Too many disturbance windows relative to total operating time.
        return False, None, None, None, None

    k_rx = p.N_d - k  # number of recrystallization disturbance windows

    def F(vars_: np.ndarray) -> np.ndarray:
        T, tau_inc = vars_
        tau_rx = p.tau_op - tau_inc
        F1 = ((tau_inc - k * p.tau_r) / p.t_inc(T) +
              k * I_inc_r(T) - 1.0)
        F2 = ((tau_rx - k_rx * p.tau_r) * p.b(T) +
              k_rx * I_rx_r(T) - p.ln_term)
        return np.array([F1, F2])

    # Initial guess from the inversion of the isothermal function.
    Tinit = invert_isothermal_T(p, T_low=p.Tmin, T_high=p.Tmax)
    tau_inc_init = p.t_inc(Tinit)
    x0 = np.array([Tinit, tau_inc_init])

    # Solve the system of equations.
    sol = root(F, x0, method='hybr', tol=tol)
    if not sol.success:
        return False, None, None, None, None

    T, tau_inc = sol.x
    tau_rx = p.tau_op - tau_inc
    F_resid = np.abs(F(np.array([T, tau_inc])))
    return True, T, tau_inc, tau_rx, F_resid


# --------------------------------------------------------------------
# 5. Helper Functions for Integer Cycle Search
# --------------------------------------------------------------------
def even_cycle_deviation(k: int, tau_inc: float, tau_rx: float, p: JMAKParams) -> float:
    """
    Computes the deviation from an ideal ratio of incubation cycles to 
    recrystallization cycles. The deviation is defined as:
    
        deviation = | (N_d_rx * tau_inc - alpha * N_d_inc * tau_rx) / (N_d_inc * tau_inc) |
    
    Args:
        k (int): Number of incubation cycles (N_d_inc).
        tau_inc (float): Incubation time.
        tau_rx (float): Recrystallization time.
        p (JMAKParams): Model parameters.

    Returns:
        float: The computed relative cycle deviation.
    """
    N_d_inc = k
    N_d_rx = p.N_d - k
    return abs((N_d_rx * tau_inc - p.alpha * N_d_inc * tau_rx) / (max(1,N_d_inc) * max(1,tau_inc)))



# --------------------------------------------------------------------
# 6. Public API: Searching for the Best Transient Solution
# --------------------------------------------------------------------
def search_solution_range(
    p: JMAKParams,
    N_low: int,
    N_high: int,
    tol_root: float = 1e-10,
    step: int = 10,
    n_best: int = 10
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Searches the integer range [N_low, N_high] (in steps of `step`) for feasible
    solutions (candidate numbers of incubation cycles). For each candidate k, the 
    function solves for T_a and tau_inc and calculates the absolute F2 residual 
    and cycle deviation.

    The outputs are sorted according to the combined normalized deviation and F2 
    residual error. The n_best candidates are returned.

    Args:
        p (JMAKParams): JMAK parameter container.
        N_low (int): Lower bound for candidate cycles.
        N_high (int): Upper bound for candidate cycles.
        tol_root (float): Tolerance for the root finder.
        step (int): Step size for candidate k values.
        n_best (int): Number of best candidates to return.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            A tuple containing arrays (candidate_N, f2, deviation, T, tau_inc) for the
            best n_best solutions, or None if no candidate is feasible.
    """
    I_inc_r, I_rx_r = single_window_integrals(p)

    f_list = []
    deviation_list = []
    candidate_N_list = []
    tau_inc_list = []
    Temp_list = []

    # Scan through candidates in steps.
    for k in range(N_low, N_high + 1, step):
        ok, T_a, tau_inc, tau_rx, F_resid = solve_Ta_tau(k, p, I_inc_r, I_rx_r, tol_root)
        # If the incubation time becomes negative, break out.
        if tau_inc is not None and tau_inc < 0:
            break
        if not ok:
            continue

        dev = even_cycle_deviation(k, tau_inc, tau_rx, p)

        candidate_N_list.append(k)
        f_list.append(F_resid.sum())
        deviation_list.append(dev)
        Temp_list.append(T_a)
        tau_inc_list.append(tau_inc)

    if not candidate_N_list:
        return None

    # Convert lists to numpy arrays.
    candidate_N = np.array(candidate_N_list)
    f = np.array(f_list)
    deviation = np.array(deviation_list)
    Temp = np.array(Temp_list)
    tau_inc = np.array(tau_inc_list)

    # Normalize deviation and f2 metrics.
    deviation_s = (deviation - deviation.min()) / (deviation.max() - deviation.min())
    f_s = (f- f.min()) / (f.max() - f.min())
    d = deviation_s + f_s
    # Sort indices based on the combined metric.
    index = np.argsort(d)
    best = index[:n_best]
    return candidate_N[best], f[best], deviation[best], Temp[best], tau_inc[best]


def best_solution_soft(
    p: JMAKParams,
    tol_root: float = 1e-10,
    cstep: int = 10,
    fstep: int = 1,
    cbest: int = 10
) -> Optional[Dict]:
    """
    Computes the best solution for the adjusted isothermal temperature T_a,
    based on a coarse search of candidate integer cycles followed by a refined search.

    The procedure is:
      1. Perform a coarse candidate search over the range [0, N_d].
      2. For each candidate from the coarse search, define a fine range around it.
      3. Select the candidate with the lowest F2 residual.

    Args:
        p (JMAKParams): JMAK parameter container.
        tol_root (float): Tolerance for the root finder.
        cstep (int): Coarse search step size.
        fstep (int): Fine search step size.
        cbest (int): Number of coarse best candidates to refine.

    Returns:
        Optional[Dict]: A dictionary with the best solution containing:
            - N_d_inc: Number of incubation cycles.
            - N_d_rx: Number of recrystallization cycles.
            - T_a: Adjusted isothermal temperature.
            - tau_inc: Incubation time.
            - tau_rx: Recrystallization time.
            - resid: Sum of absolute residual.
            - cycle_dev: Cycle deviation.
        If no feasible candidate exists, returns None.
    """
    # Coarse search for candidates.
    coarse_search = search_solution_range(p, 0, p.N_d, tol_root=tol_root, step=cstep, n_best=cbest)
    if coarse_search is None:
        raise ValueError('Coarse search failed: no candidate solution found.')

    candidates, _, _, _, _ = coarse_search

    # Define fine search ranges for each coarse candidate.
    fine_ranges = [
        (max(0, int(k) - cstep), min(p.N_d, int(k) + cstep))
        for k in candidates
    ]

    best_sol = {
        'N_d_inc': None,
        'resid': float('inf'),
        'cycle_dev': float('inf')
    }

    for fr in fine_ranges:
        fine_search = search_solution_range(p, fr[0], fr[1], tol_root=tol_root, step=fstep, n_best=1)
        if fine_search is None:
            continue
        candidate_arr, resid_arr, dev_arr, Ta_arr, tau_inc_arr = fine_search
        # Update if this candidate has a lower F2 residual.
        if resid_arr[0] < best_sol['resid']:
            best_sol['N_d_inc'] = int(candidate_arr[0])
            best_sol['resid'] = resid_arr[0]
            best_sol['cycle_dev'] = dev_arr[0]
            best_sol['T_a'] = Ta_arr[0]
            best_sol['tau_inc'] = tau_inc_arr[0]

    if best_sol['N_d_inc'] is None:
        return None
    else:
        best_sol['N_d_rx'] = p.N_d - best_sol['N_d_inc']
        best_sol['tau_rx'] = p.tau_op - best_sol['tau_inc']
        return best_sol


# --------------------------------------------------------------------
# Example Usage / Testing
# --------------------------------------------------------------------
if __name__ == "__main__":
    # This example demonstrates how to define parameters and run the solver.
    # Note: In an actual deployment or testing framework, replace these with
    # your measured or estimated parameters and temperature profiles.
    
    # Define a dummy time grid and transient temperature profile.
    Nt = 100
    t_array = np.linspace(0, 10, Nt)
    # Example: a sinusoidal temperature disturbance with small amplitude.
    deltaT = 5.0 * np.sin(np.linspace(0, np.pi, Nt))

    # Create a JMAKParams instance with example parameters.
    params = JMAKParams(
        n=2.0,
        a1=10.0,
        B1=5000.0,
        a2=8.0,
        B2=3000.0,
        X_star=0.8,
        tau_op=1000.0,   # operating time [s]
        tau_r=10.0,      # disturbance window duration [s]
        N_d=20,          # total disturbance windows
        alpha=1.0,       # cycle ratio parameter (example)
        deltaT_r=deltaT,
        t=t_array,
        Tmin=600.0,
        Tmax=4000.0
    )

    # Invert the isothermal function to get T_iso (for reference).
    T_iso = invert_isothermal_T(params)
    print("Inverted isothermal temperature T_iso =", T_iso)

    # Solve for the best transient solution using the provided API.
    best_solution = best_solution_soft(params)
    if best_solution is not None:
        print("Best solution for transient events:")
        for key, value in best_solution.items():
            print(f"  {key}: {value}")
    else:
        print("No feasible transient solution found.")
