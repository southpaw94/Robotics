from numpy import cross, dot, matrix, arcsin, sin, cos, transpose, arccos, trace
from numpy.linalg import norm

def skew(k_norm):
    return matrix([[0, -k_norm[2], k_norm[1]], \
            [k_norm[2], 0, -k_norm[0]], \
            [-k_norm[1], k_norm[0], 0]])

def vec_to_r(v1, v2):
    """
    Parameters
    ----------
    v1: array-like (1x3)
        First 3D vector
    v2: array-like (1x3)
        Second 3D vector

    Returns
    -------
    3x3 numpy matrix representing the rotation matrix R
    which will satisfy the equation "R*v1 = v2"

    """

    k = cross(v1, v2)
    k_norm = k / norm(k)

    K = skew(k_norm)

    phi = arcsin(norm(k) / (norm(v1) * norm(v2)))

    I = matrix([[1, 0, 0], \
            [0, 1, 0], \
            [0, 0, 1]])

    R = I + K * sin(phi) + K**2 * (1 - cos(phi))
    return R

def r_to_kphi(R):
    """
    Parameters
    ----------
    R: numpy matrix
        nxn rotation matrix

    Returns
    -------
    array-like object representing single rotation vector
    single axis rotation angle phi in radians

    """

    phi = arccos((trace(R) - 1) / 2)
    K = (R - R.T) / (2*sin(phi))
    k = [-K[1, 2], K[0, 2], -K[0,1]]
    return k, phi

def kphi_to_r(k, phi):
    """
    Parameters
    ----------
    k: array-like
        rotation vector about which the frame is rotated
    phi: float
        degree of rotation in radians

    Returns
    -------
    single rotation matrix
    """
    
    k = k / norm(k)
    I = matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    K = skew(k)

    R = I + K * sin(phi) + K**2 * (1 - cos(phi))

    return R

def euler_to_r(phi, theta, psi):
    """
    Parameters
    ----------
    phi: float
        degree of roll rotation in radians
    theta: float
        degree of pitch rotation in radians
    psi: float
        degree of yaw rotation in radians

    Returns
    -------
    R: array-like
        Resulting 3x3 rotation matrix
    """

    R_roll = matrix([[cos(phi), -sin(phi), 0], \
            [sin(phi), cos(phi), 0], \
            [0, 0, 1]])

    R_pitch = matrix([[cos(theta), 0, sin(theta)], \
            [0, 1, 0], \
            [-sin(theta), 0, cos(theta)]])

    R_yaw = matrix([[1, 0, 0], \
            [0, cos(psi), -sin(psi)], \
            [0, sin(psi), cos(psi)]])

    return R_roll * R_pitch * R_yaw
