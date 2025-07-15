import numpy as np
from def_orbites import OrbiteROE, OrbiteFactory

def get_reference_ROE(self, num_points=20):
    points = []

    for theta_deg in np.linspace(0, 360, num_points):
        roe_step = [self.x_r, self.y_r, self.a_r, theta_deg, self.A_z, self.gamma]
        points.append(roe_step)
    
    return np.array(points)


def initialize_parts_visited(reference_points_by_orbit):
    """
    Initialise les points visités à False pour chaque orbite.

    :param reference_points_by_orbit: dict {orbit_id: [roe_point_1, ...]}
    :return: dict {orbit_id: [(False, roe_point)...}
    """
    parts_visited_on_orbits = {
        orbit_id: [(False, roe_point) for roe_point in points]
        for orbit_id, points in reference_points_by_orbit.items()
    }
    return parts_visited_on_orbits
