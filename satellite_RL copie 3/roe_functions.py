import numpy as np

from hcw_propagator import propagate

#import utils as utl

def state2roe(rel_state, n, z_roe1='A_z', z_roe2='gamma'):
    
    x, y, z = rel_state[0]
    vx, vy, vz = rel_state[1]

    x_r = 4 * x + 2 * vy / n
    y_r = y - 2 * vx / n
    a_r = np.sqrt((6 * x + 4 * vy / n) ** 2 + (2 * vx / n) ** 2)
    E_r = np.rad2deg(np.arctan2(2 * vx / n, 6 * x + 4 * vy / n))
    A_z = np.sqrt(z ** 2 + (vz / n) ** 2)
    gamma = (np.rad2deg(np.arctan2(z, vz / n)) - E_r)
    psi = np.rad2deg(np.arctan2(z, vz / n))
    i_r = np.arccos(-a_r / np.sqrt(A_z ** 2 * (4 * np.sin(np.deg2rad(gamma)) ** 2 + np.cos(np.deg2rad(gamma)) ** 2) + a_r ** 2))

    if z_roe1 == 'A_z':
        roe1 = A_z
    elif z_roe1 == 'i_r':
        roe1 = np.rad2deg(i_r)
    else:
        raise ValueError('Choose one between A_z and i_r.')

    if z_roe2 == 'psi':
        roe2 = psi
    elif z_roe2 == 'gamma':
        roe2 = gamma
    else:
        raise ValueError('Choose one between psi and gamma.')

    return [x_r, y_r, a_r, E_r, roe1, roe2]







def inc2zamp(sat_roe, z_roe2='gamma'):
    _, _, a_r, E_r = sat_roe[0], sat_roe[1], sat_roe[2], sat_roe[3]

    if z_roe2 == 'psi':
        psi = sat_roe[5]
        gamma = (psi - E_r)
    elif z_roe2 == 'gamma':
        gamma = sat_roe[5]
    else:
        raise ValueError('Choose one between psi and gamma.')

    i_r = sat_roe[4]
    A_z = a_r * np.sqrt((1 / (np.cos(np.deg2rad(i_r))) ** 2 - 1) / (4 * (np.sin(np.deg2rad(gamma))) ** 2 + (np.cos(np.deg2rad(gamma))) ** 2))
    return A_z






def zamp2inc(sat_roe, z_roe2='gamma'):
    _, _, a_r, E_r = sat_roe[0], sat_roe[1], sat_roe[2], sat_roe[3]

    if z_roe2 == 'psi':
        psi = sat_roe[5]
        gamma = psi - E_r
    elif z_roe2 == 'gamma':
        gamma = sat_roe[5]
    else:
        raise ValueError('Choose one between psi and gamma.')

    A_z = sat_roe[4]
    i_r = np.rad2deg(np.arccos(-a_r / np.sqrt(A_z ** 2 * (4 * np.sin(np.deg2rad(gamma)) ** 2 + np.cos(np.deg2rad(gamma)) ** 2) + a_r ** 2)))

    return i_r





def psi2gamma(sat_roe):
    E_r = sat_roe[3]
    psi = sat_roe[5]

    gamma = psi - E_r

    return gamma



def gamma2psi(sat_roe):
    E_r = sat_roe[3]
    gamma = sat_roe[5]

    psi = gamma + E_r

    return psi






def roe2state(sat_roe, n, z_roe1='A_z', z_roe2='gamma'):
    x_r, y_r, a_r, E_r = sat_roe[0], sat_roe[1], sat_roe[2], sat_roe[3]

    if z_roe2 == 'psi':
        psi = sat_roe[5]
        gamma = (psi - E_r)
    elif z_roe2 == 'gamma':
        gamma = sat_roe[5]
        psi = (E_r + gamma)
    else:
        raise ValueError('Choose one between psi and gamma.')

    if z_roe1 == 'A_z':
        A_z = sat_roe[4]
    elif z_roe1 == 'i_r':
        i_r = sat_roe[4]
        A_z = a_r * np.sqrt((1 / (np.cos(np.deg2rad(i_r))) ** 2 - 1) / (4 * (np.sin(np.deg2rad(gamma))) ** 2 + (np.cos(np.deg2rad(gamma))) ** 2))
    else:
        raise ValueError('Choose one between A_z and i_r.')

    x = x_r - 0.5 * a_r * np.cos(np.deg2rad(E_r))
    y = y_r + a_r * np.sin(np.deg2rad(E_r))
    z = A_z * np.sin(np.deg2rad(psi))

    x_dot = 0.5 * n * a_r * np.sin(np.deg2rad(E_r))
    y_dot = -3 * n * x_r / 2 + n * a_r * np.cos(np.deg2rad(E_r))
    z_dot = n * A_z * np.cos(np.deg2rad(psi))

    return [np.array([x, y, z]), np.array([x_dot, y_dot, z_dot])]


def orbit_from_roe(sat_roe, n, num_orbits, z_roe1='A_z', z_roe2='gamma'):
    T_orbit = 2 * np.pi / n
    T_total = num_orbits * T_orbit


    sat_roe[3] = 0

    state0 = roe2state(sat_roe, n, z_roe1=z_roe1, z_roe2=z_roe2)
    t_span = [0, T_total]
    t_eval = np.linspace(0, T_total, 1000 * num_orbits)
    _, orbit_state = propagate(state0, t_span, n, lin_ctrl=np.zeros(3), t_eval=t_eval)

    return orbit_state


def angular_difference(angle1, angle2):
    return (angle2 - angle1 + 180) % 360 - 180



def roe_difference(roe1, roe2, spatial_weight=1.0, angular_weight=1.0, z_roe1='A_z', z_roe2='gamma'):
    """
    Computes a weighted norm between two ROE vectors.

    :param roe1: List of 6 elements representing the first ROE vector [x, y, a, E, i, gamma],
                 where x, y, a are distances and E, i, gamma are angles in degrees.
    :param roe2: List of 6 elements representing the second ROE vector [x, y, a, E, i, gamma],
                 where x, y, a are distances and E, i, gamma are angles in degrees.
    :param spatial_weight: Weight applied to the spatial norm component (default is 1.0).
    :param angular_weight: Weight applied to the angular norm component (default is 1.0).
    :return: A float representing the combined weighted norm between the two ROE vectors.
    """
    if len(roe1) != 6 or len(roe2) != 6:
        raise ValueError("Both ROE vectors must have 6 elements: [x, y, a, E, i, gamma]")

    # Spatial differences (Euclidean)
    dx = roe2[0] - roe1[0]
    dy = roe2[1] - roe1[1]
    da = roe2[2] - roe1[2]

    # Angular differences
    if (roe1[3] is not None and roe2[3] is not None):
        dE = angular_difference(roe1[3], roe2[3])
    else:
        dE = 0

    dgamma = angular_difference(roe1[5], roe2[5])

    if z_roe1 == 'A_z':
        dA = roe2[4] - roe1[4]
        spatial_norm = np.sqrt(dx ** 2 + dy ** 2 + da ** 2 + dA ** 2)
        angular_norm = np.sqrt(dE ** 2 + dgamma ** 2)
    elif z_roe1 == 'i_r':
        #di = utl.angular_difference(roe1[4], roe2[4])
        spatial_norm = np.sqrt(dx ** 2 + dy ** 2 + da ** 2)
        angular_norm = np.sqrt(dE ** 2 + di ** 2 + dgamma ** 2)
    else:
        raise ValueError('Choose one between A_z and i_r.')

    # Combined weighted norm
    total_norm = spatial_weight * spatial_norm + angular_weight * angular_norm

    return total_norm















def alt2default_roes(sat_roe_alt):
    x_r, y_r, a_r, E_r, i_r, gamma = sat_roe_alt

    A_z = a_r * np.sqrt((1 / (np.cos(np.deg2rad(i_r))) ** 2 - 1) / (4 * (np.sin(np.deg2rad(gamma))) ** 2 + (np.cos(np.deg2rad(gamma))) ** 2))
    psi = (E_r + gamma)

    return [x_r, y_r, a_r, E_r, A_z, psi]



