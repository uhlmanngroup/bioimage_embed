import numpy as np
from scipy.interpolate import interp1d, splprep, splev


def cart2pol(x, y):
    return (np.sqrt(x**2 + y**2), np.arctan2(y, x))


def pol2cart(rho, phi):
    return (rho * np.cos(phi), rho * np.sin(phi))


def cubic_polar_resample_contour(contour: np.array, size: int) -> np.array:
    """Star convex resampling of a contour using cubic interpolation

    Args:
        contour (np.Array): scikit image contour
        size (int): control points to interpolate to

    Returns:
        np.Array: new contour
    """
    contour_y, contour_x = contour[0][:, 0], contour[0][:, 1]
    rho, phi = cart2pol(contour_x, contour_y)

    rho_interp = interp1d(np.linspace(0, 1, len(rho)), rho, kind="cubic")(
        np.linspace(0, 1, size)
    )
    phi_interp = interp1d(np.linspace(0, 1, len(phi)), phi, kind="cubic")(
        np.linspace(0, 1, size)
    )

    xii, yii = pol2cart(rho_interp, phi_interp)
    return np.array([xii, yii])


def contour_to_xy(contour: np.array):
    return contour[0][:, 0], contour[0][:, 1]


def uniform_spline_resample_contour(contour: np.array, size: int) -> np.array:
    """Resample a contour using a uniform spline
    Author: @afoix

    Args:
        contour (np.array): scikit image contour
        size (int): Control points to interpolate to

    Returns:
        np.Array: new contour
    """
    contour_y, contour_x = contour_to_xy(contour)
    tck, u = splprep([contour_x, contour_y], s=0)
    u_new = np.linspace(u.min(), u.max(), size)
    return np.array(splev(u_new, tck))
