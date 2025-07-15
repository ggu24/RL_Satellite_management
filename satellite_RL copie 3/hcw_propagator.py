import numpy as np
from scipy.integrate import solve_ivp, odeint

def hcw_eq(t, state, sat_n, lin_ctrl):
    """
    Computes the time derivative of the state vector using the Hill-Clohessy-Wiltshire (HCW) equations of relative motion.

    :param t: Current time.
    :param array_like state: Current state vector [x, y, z, vx, vy, vz].
    :param float sat_n: Mean motion of the reference orbit (rad/s).
    :param array_like lin_ctrl: Optional control acceleration vector [ax, ay, az].
    :return: Time derivative of the state vector [vx, vy, vz, ax, ay, az].
    :rtype: ndarray
    """
    r = state[0:3]
    v = state[3:6]
    # x, y, z, vx, vy, vz = state

    x = r[0]
    y = r[1]
    z = r[2]
    vx = v[0]
    vy = v[1]
    vz = v[2]

    # Orbital dynamics (Hill-Clohessy–Wiltshire)
    x_dot_dot = 2*sat_n*vy + 3*sat_n**2 *x
    y_dot_dot = - 2*sat_n*vx
    z_dot_dot = - sat_n**2 * z

    acc = np.array([x_dot_dot, y_dot_dot, z_dot_dot])

    # Add linear control acceleration
    if lin_ctrl is not None:
        acc += lin_ctrl



    # Retrieve the derivative of the state vector
    dot_state = np.zeros_like(state)
    dot_state[0:3] = v
    dot_state[3:6] = acc

    return dot_state


def propagate(state0, t_span, sat_n, lin_ctrl=None, method='RK45', rtol=1e-10, atol=1e-20, t_eval=None):
    """
    Propagate the relative motion of a satellite using the HCW equations.

    :param list state0: Initial state as [position (3,), velocity (3,)].
    :param tuple t_span: Time span for integration (start_time, end_time).
    :param float sat_n: Mean motion of the reference orbit (rad/s).
    :param array_like lin_ctrl: Optional constant control acceleration [ax, ay, az].
    :param str method: Integration method (default: 'RK45').
    :param float rtol: Relative tolerance for the integrator.
    :param float atol: Absolute tolerance for the integrator.
    :param array_like t_eval: Specific time points to evaluate the solution.
    :return: Tuple containing:
        - t (ndarray): Evaluation time points.
        - state (list): [positions (n, 3), velocities (n, 3)]
    :rtype: tuple
    """
    state0_concatenate = np.concatenate((state0[0], state0[1]))


    sol = solve_ivp(hcw_eq, t_span, state0_concatenate, args=(sat_n, lin_ctrl), method=method, rtol=rtol, atol=atol,
                    t_eval=t_eval)



    # Prepare the output based on t_eval
    if t_eval is not None and len(t_eval) == 2:
            #if t_eval == t_span:
            if tuple(t_eval) == t_span:
                state = [sol.y.T[1, 0:3], sol.y.T[1, 3:6]]
            else:
                state = [sol.y.T[:, 0:3], sol.y.T[:, 3:6]]
            



    return sol.t, state



