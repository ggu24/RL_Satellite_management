import math as mt
import numpy as np

import roe_functions as roe

########################################################################################################################
###################################### ARTIFICIAL POTENTIAL FIELDS CONTROLLERS #########################################
########################################################################################################################


############################################### RENEVEY FORMULATION ####################################################

def apf_renevey(sat_state, sat_target, sat_n, deltaT, old_deltaV2, ka=1.33e-02, deltaV_min=1e-8, deltaV_max=1e3, return_error= True):
    '''
    Computes the control commands for a satellite using the Artificial Potential Fields (APF) method
    in the formulation described by Renevey. The function updates the satellite's relative orbital elements (ROEs)
    and calculates the delta-V commands to drive the satellite towards the target.

    :param sat_state: Current relative state of the satellite [position, velocity] in Cartesian coordinates.
    :param sat_target: Targeted ROEs [x_r, y_r, a_r, E_r, A_z, gamma].
    :param sat_n: Mean motion of the satellite.
    :param deltaT: Time interval between control updates.
    :param old_deltaV2: Previous iteration's deltaV2 (velocity update from the previous step).
    :param ka: APF gain for the control law (default 1.33e-02).
    :param deltaV_min: Minimum threshold for delta-V magnitude (default 1e-16).
    :param deltaV_max: Maximum threshold for delta-V magnitude (default 1e3).
    :return: A tuple (deltaV1, deltaV2) where deltaV1 is the immediate velocity correction,
             and deltaV2 is the velocity correction for the next iteration.
    '''

    # Si on a un intervalle [t0, t1], on prend la durée
    if isinstance(deltaT, (list, np.ndarray)) and len(deltaT) == 2:
        deltaT = float(deltaT[1] - deltaT[0])
    else:
        deltaT = float(deltaT)



    # Compute the virtual state with deltaV2 of the last cycle applied
    sat_virtual_state = [sat_state[0], sat_state[1] + old_deltaV2]

    # Satellite ROEs
    [x_r, y_r, a_r, E_r, A_z, gamma] = roe.state2roe(sat_virtual_state, sat_n, z_roe1='A_z', z_roe2='gamma')

    # Extract targeted ROEs
    [x_r_t, y_r_t, a_r_t, E_r_t, A_z_t, gamma_t] = sat_target

    x_r = np.array(x_r)
    x_r_t = np.array(x_r_t)

    # Compute current error
    error = np.sqrt((x_r - x_r_t)**2 + (y_r - y_r_t)**2 + (a_r - a_r_t)**2 + (A_z - A_z_t)**2 + np.deg2rad((gamma - gamma_t + 180) % 360 - 180)**2)

    # Compute updated relative orbital elements with apf control. It will be the next targeted point in cartesian coordinates
    x_r_p = x_r - ka * (x_r - x_r_t) * deltaT
    y_r_p = y_r - 1.5 * sat_n * x_r * deltaT - ka * (y_r - y_r_t) * deltaT
    a_r_p = a_r - ka * (a_r - a_r_t) * deltaT
    E_r_p = E_r + np.rad2deg(sat_n * deltaT)
    if A_z_t is not None and gamma_t is not None:
        A_z_p = A_z - ka * (A_z - A_z_t) * deltaT
        gamma_p = gamma - ka * ((gamma - gamma_t + 180) % 360 - 180) * deltaT
    else:
        A_z_p = A_z
        gamma_p = gamma

    #if E_r_t is not None:
    #    E_r_p -= ka * (E_r - E_r_t) * deltaT
    if E_r_t is not None:
        diff_Er = (E_r - E_r_t + 180) % 360 - 180  # wrap-around to [-180, 180]
        E_r_p -= ka * diff_Er * deltaT

    # Write the ROEs of the new waypoint to follow
    threshold = 1e-6
    new_roe = np.array([x_r_p, y_r_p, a_r_p, E_r_p, A_z_p, gamma_p])
    new_roe[np.isclose(new_roe, 0, atol=threshold)] = 0
    new_roe = list(new_roe)

    # Transform the targeted waypoint roe into a cartesian coordinates state
    new_state = roe.roe2state(new_roe, sat_n, z_roe1='A_z', z_roe2='gamma')

    # Retrieve State Transition Matrix
    stm = get_hcw_stm(sat_n, deltaT)
    stm_11 = stm[:3, :3]
    stm_12 = stm[:3, 3:6]
    stm_21 = stm[3:6, :3]
    stm_22 = stm[3:6, 3:6]

    stm_12_inv = get_hcw_stm_12_inv(sat_n, deltaT)

    # Compute deltaV in order to reach the targeted cartesian state
    deltaV = stm_12_inv @ (new_state[0] - (stm_11@sat_state[0] + stm_12@sat_state[1]))

    # Compute deltaV2 that should be applied at next iteration
    deltaV2 = new_state[1] - stm_21 @ sat_state[0] - stm_22 @ (sat_state[1] + deltaV)

    # Compute the magnitude of the deltaV vector
    deltaV_mag = np.linalg.norm(deltaV)

    # Check if the magnitude is below deltaV_min
    if deltaV_mag < deltaV_min:
        if np.linalg.norm(old_deltaV2) < deltaV_min:
            deltaV1 = np.zeros(3)
            deltaV2 = np.zeros(3)
        else:
            deltaV1 = old_deltaV2
            deltaV2 = np.zeros(3)

    # Check if the magnitude is above deltaV_max
    elif deltaV_mag > deltaV_max:
        # Scale the vector to have magnitude deltaV_max
        deltaV1 = deltaV * (deltaV_max / deltaV_mag)

    # Otherwise return the computed deltaV
    else:
        deltaV1 = deltaV

    if return_error:
        return [deltaV1, deltaV2], error
    else:
        return deltaV1, deltaV2



def get_hcw_stm(n, deltaT):
    '''
    Computes the State Transition Matrix (STM) for the Hill-Clohessy-Wiltshire (HCW) equations.

    :param n: Mean motion of the satellite.
    :param deltaT: Time interval for the STM computation.
    :return: A 6x6 STM matrix that maps the state transition under HCW dynamics.
    '''

    #print("🔍 get_hcw_stm — type(deltaT):", type(deltaT), "value:", deltaT)

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