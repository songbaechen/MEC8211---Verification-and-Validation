import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np

from problem_definition.lbm_devoir3 import Generate_sample, LBM
from problem_definition.problem_definition import SampleParameters, ProblemParameters
from plot_convergence import plot_convergence_and_gci


# def richardson_convergence(
#         srq_list: list[float],
#         dx_list: list[float],
#         p_est: float = 3.0,
#         max_n_iter: int = 100,
#         tol: float = 5e-3
#     ) -> float:
#     """
#     1: finest mesh
#     2: normal/median mesh
#     3: gross mesh
#     """
#     r12 = dx_list[-1] / dx_list[-2]
#     r23 = dx_list[-2] / dx_list[-3]
#
#     f1 = srq_list[-1]
#     f2 = srq_list[-2]
#     f3 = srq_list[-3]
#
#     # initial estimation of p
#     p_k = p_est
#
#     for i in range(max_n_iter):
#         up_term = (r12 ** p_k - 1) * (f3 - f2) / (f2 - f1) + r12 ** p_k
#         p_kp1 = np.log(up_term) / np.log(r12 * r23)
#
#         if abs(p_kp1 - p_k) < tol:
#             return p_kp1
#
#         p_k = p_kp1
#
#     raise ValueError("Richardson iterative method did not converge (inside fct 'richardson_convergence')'")
#
#
# def gci_factor(
#         srq_list: list[float],
#         fx_list: list[float],
#         p_hat: float,
#         p_formal: float = 3.0
#     ) -> float:
#     """
#     1: finest mesh
#     2: gross mesh
#     """
#     f1 = srq_list[-1]
#     f2 = srq_list[-2]
#     r = fx_list[-1] / fx_list[-2]
#
#     p_ratio = abs( (p_hat - p_formal) / p_formal )
#
#     if p_ratio > 0.1:
#         f_s = 3.0
#         p = min( max(0.5, p_hat), p_formal )
#
#     else:
#         f_s = 1.25
#         p = p_formal
#
#     return f_s * abs(f2 - f1) / (r ** p - 1)


def convergence_analysis(
        initial_dx: float,
        initial_nx: float,
        ratio: float,
        n_steps: int,
        sample_parameters: SampleParameters,
        problem_parameters: ProblemParameters
    ) -> tuple(list[float], list[float]):
    """
    Keeping nx * dx constant by using a ratio
    """
    k_list = []
    dx_list = []
    for i in range(n_steps):
        dx = initial_dx / (i + 1)
        nx = initial_nx * (i + 1)

        d_equivalent = Generate_sample(
            seed=sample_parameters.seed,
            filename=sample_parameters.filename,
            mean_d=sample_parameters.mean_d,
            std_d=sample_parameters.std_d,
            poro=sample_parameters.poro,
            nx=nx,
            dx=dx
        )
        k = LBM(
            filename=problem_parameters.filename,
            NX=nx,
            deltaP=problem_parameters.deltaP,
            dx=dx,
            d_equivalent=d_equivalent
        )

        k_list.append(k)
        dx_list.append(dx)

    return k_list, dx_list


if __name__ == "__main__":
    filename = "convergence_space.tiff"
    nx = 100
    dx = 2e-5

    sample_parameters = SampleParameters(
        seed=0,
        filename=filename,
        mean_d=12.5,
        std_d=2.85,
        poro=0.9,
        nx=nx,
        dx=dx
    )
    problem_parameters = ProblemParameters(
        filename=filename,
        NX=nx,
        deltaP=0.1,
        dx=dx,
        d_equivalent=0.0        # temporary will be changed inside function
    )
    k_list, dx_list = convergence_analysis(
        initial_dx=dx,
        initial_nx=nx,
        ratio=2,
        n_steps=3,
        sample_parameters=sample_parameters,
        problem_parameters=problem_parameters
    )

    plot_convergence_and_gci(k_list, dx_list, p_formal=2.0)

