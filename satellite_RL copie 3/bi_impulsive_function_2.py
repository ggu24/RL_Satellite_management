import numpy as np
from scipy.optimize import minimize



def bi_impulsive_transfer(n, sat_state0, sat_roe1, initial_guess):
    '''
    Computes the optimal in-plane eccentricity vector (E_r) and time of flight (deltaT)
    that minimize the total delta-V required for a two-impulse rendezvous using
    the Hill-Clohessy-Wiltshire (HCW) dynamics model.

    The optimization minimizes the sum of two velocity changes (deltaV1 and deltaV2),
    subject to reaching a final relative orbital element (ROE) state.

    :param n: Mean motion of the chief satellite [rad/s]
    :param sat_state0: Tuple of (r0, r_dot0_minus)
                       r0: initial relative position vector in LVLH frame [3x1]
                       r_dot0_minus: relative velocity just before first impulse [3x1]
    :param sat_roe1: Final desired relative orbital elements (ROE) [6x1]
    :param initial_guess: Initial guess for the optimizer [E_r (3x1), deltaT (scalar)]
    :return: A dictionary containing:
             - 'deltaV1': First impulse delta-V vector [3x1]
             - 'deltaV2': Second impulse delta-V vector [3x1]
             - 'deltaT': Time of flight between impulses [scalar]
             - 'E_r': Optimal final in-plane eccentricity vector [3x1]
             - 'total_deltaV': Sum of norms of deltaV1 and deltaV2
    '''

    def compute_final_state(n, sat_roe1, E_r):
        x_r = sat_roe1.x_r
        y_r = sat_roe1.y_r
        a_r = sat_roe1.a_r
        A_z = sat_roe1.A_z
        gamma = sat_roe1.gamma
        gamma_rad = np.deg2rad(gamma)
        E_r_rad = np.deg2rad(E_r)
        r1 = np.array([
            x_r - 0.5 * a_r * np.cos(E_r_rad),
            y_r + a_r * np.sin(E_r_rad),
            A_z * np.sin(E_r_rad + gamma_rad)
        ])
        r_dot1_plus = np.array([
            0.5 * n * a_r * np.cos(E_r_rad),
            -3 * 0.5 * n * x_r + n * a_r * np.cos(E_r_rad),
            n * A_z * np.cos(E_r_rad + gamma_rad)
        ])
        return r1, r_dot1_plus


    def deltaV_minimizer(vars, n, sat_state0, sat_roe1):

        E_r = vars[:3]
        deltaT = vars[3]


        stm = get_hcw_stm(n, deltaT)

        # Extract submatrices
        stm_11 = stm[:3, :3]
        stm_12 = stm[:3, 3:6]
        stm_21 = stm[3:6, :3]
        stm_22 = stm[3:6, 3:6]

        stm12_inv = get_hcw_stm_12_inv(n, deltaT)

        # Compute r1 and r_dot1_plus from final ROE and in-plane vector
        r1, r_dot1_plus = compute_final_state(n, sat_roe1, E_r)

        # Initial conditions
        r0, r_dot0_minus = sat_state0

        # Delta-V1 cost
        term1 = np.linalg.norm(stm12_inv @ (r1 - stm_11 @ r0) - r_dot0_minus)

        # Delta-V2 cost
        term2 = np.linalg.norm(r_dot1_plus - (stm_21 - stm_22 @ stm12_inv @ stm_11) @ r0 - stm_22 @ stm12_inv @ r1)

        return term1 + term2

    # Run optimization
    result = minimize(
        deltaV_minimizer,
        initial_guess,
        args=(n, sat_state0, sat_roe1),
        method='Nelder-Mead'
    )

    # Extract optimized variables
    optimized_vars = result.x
    E_r_opt = optimized_vars[:3]
    deltaT_opt = optimized_vars[3]

    # Recompute STM and states
    stm = get_hcw_stm(n, deltaT_opt)
    stm_11 = stm[:3, :3]
    stm_12 = stm[:3, 3:6]
    stm_21 = stm[3:6, :3]
    stm_22 = stm[3:6, 3:6]
    stm12_inv = get_hcw_stm_12_inv(n, deltaT_opt)

    r0, r_dot0_minus = sat_state0
    r1, r_dot1_plus = compute_final_state(n, sat_roe1, E_r_opt)

    print("r0:", r0.shape)
    print("r_dot0_minus:", r_dot0_minus.shape)
    print("r1:", r1.shape)
    print("r_dot1_plus:", r_dot1_plus.shape)
    print("stm_11:", stm_11.shape)
    print("stm_12:", stm_12.shape)
    print("stm12_inv:", stm12_inv.shape)
    print("stm_21:", stm_21.shape)
    print("stm_22:", stm_22.shape)

    # Compute delta-Vs
    deltaV1 = stm12_inv @ (r1 - stm_11 @ r0) - r_dot0_minus
    deltaV2 = r_dot1_plus - (stm_21 - stm_22 @ stm12_inv @ stm_11) @ r0 - stm_22 @ stm12_inv @ r1
    #total_deltaV = np.linalg.norm(deltaV1) + np.linalg.norm(deltaV2)

    return [deltaV1, deltaV2, deltaT_opt, E_r_opt]




def get_hcw_stm(n, deltaT):
    '''
    Computes the State Transition Matrix (STM) for the Hill-Clohessy-Wiltshire (HCW) equations.

    :param n: Mean motion of the satellite.
    :param deltaT: Time interval for the STM computation.
    :return: A 6x6 STM matrix that maps the state transition under HCW dynamics.
    '''

    stm = np.array([[4-3*np.cos(n*deltaT), 0, 0, np.sin(n*deltaT)/n, 2*(1-np.cos(n*deltaT))/n, 0],
                    [6*(-n*deltaT+np.sin(n*deltaT)), 1, 0, 2*(np.cos(n*deltaT)-1)/n, -3*deltaT + 4*np.sin(n*deltaT)/n, 0],
                    [0, 0, np.cos(n*deltaT), 0, 0, np.sin(n*deltaT)/n],
                    [3*n*np.sin(n*deltaT), 0, 0, np.cos(n*deltaT), 2*np.sin(n*deltaT), 0],
                    [6*n*(np.cos(n*deltaT)-1), 0, 0, -2*np.sin(n*deltaT), -3+4*np.cos(n*deltaT), 0],
                    [0, 0, -n*np.sin(n*deltaT), 0, 0, np.cos(n*deltaT)]])

    return stm


def get_hcw_stm_12_inv(n, deltaT):
    '''
    Computes the inverse of the STM sub-matrix (STM_12) for the HCW equations. This sub-matrix is
    used to compute the velocity corrections required to reach a target state.

    :param n: Mean motion of the satellite.
    :param deltaT: Time interval for the STM computation.
    :return: A 3x3 inverse STM_12 matrix.
    '''

    frac = 8 * (np.cos(n * deltaT) - 1) + 3 * n * deltaT * np.sin(n * deltaT)
    stm_12_inv = np.array([[(3*n**2 *deltaT - 4*n*np.sin(n*deltaT))/frac, 2*n*(1-np.cos(n*deltaT))/frac, 0],
                           [2*n*(np.cos(n*deltaT)-1)/frac, -n*np.sin(n*deltaT)/frac, 0],
                           [0, 0, n/np.sin(n*deltaT)]])

    return stm_12_inv

