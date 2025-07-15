import numpy as np
import matplotlib.pyplot as plt
from hcw_propagator import propagate
from roe_functions import roe2state  # ou roe2position selon ton code
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass



y_r_constant = False ### To tell if you want a constant y_r (Very important for the training results)


@dataclass
class OrbiteROE:
    x_r: float     # radial displacement (m)
    y_r: float     # along-track displacement (m)
    a_r: float     # semi-major axis difference (m)
    E_r: float     # relative eccentricity phase (deg)
    A_z: float     # cross-track amplitude (m)
    gamma: float   # phase of inclination vector (deg)

    def get_label(self):
        return f"x_r={self.x_r:.0f}m, y_r={self.y_r:.0f}m, a_r={self.a_r:.0f}m, E_r={self.E_r:.0f}°, A_z={self.A_z:.0f}m"
    
    def get_reference_points(self, sat_n, num_points=30):
        points = []

        for theta_deg in np.linspace(0, 360, num_points):
            roe_step = [self.x_r, self.y_r, self.a_r, theta_deg, self.A_z, self.gamma]
            pos, _ = roe2state(roe_step, sat_n, z_roe1='A_z', z_roe2='gamma')
            points.append(pos)
        
        return np.array(points)




class OrbiteFactory:
    def __init__(self, N=5, rayon_meteorite=10000): #### HERE TO CHANGE THE NUMBER OF ORBITS N
        self.N = N
        self.rayon = rayon_meteorite
        self.orbites = self.generate_random_orbits()

    def generate_random_orbits(self):
        orbites = []
        for i in range(self.N):
            x_r = 0

            if y_r_constant :
                y_r = 5      # few meters (pour la plupart des tests)
            else :
                y_r = 3 - (self.N - i)

            a_r = 10     # few meters 
            E_r = (self.N - i) / self.N * 360 # Full eccentric anomaly range
            A_z = (self.N - i) / self.N * 10
            gamma = (self.N - i) / self.N * 360 # phase angle
            orbites.append(OrbiteROE(x_r, y_r, a_r, E_r, A_z, gamma))
        return orbites


    def plot_summary_orbits(self, sat_n, num_points=500, show_reference_points=True):

        """
        Trace chaque orbite comme une trajectoire 3D dérivée des ROEs + éventuellement les points de référence.

        :param sat_n: mean motion (rad/s)
        :param num_points: nombre de points sur chaque orbite
        :param show_reference_points: booléen pour activer l’affichage des points clés
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Orbites générées autour de l’astéroïde (via ROEs)")


        # Rayon de l'astéroïde (en mètres, adapte selon ton cas)
        r_ast = 1

        # Création d'une sphère en 3D
     
        x0, y0, z0 = 0, 0, 0
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

        x_sphere = x0 + r_ast * np.cos(u) * np.sin(v)
        y_sphere = y0 + r_ast * np.sin(u) * np.sin(v)
        z_sphere = z0 + r_ast * np.cos(v)

        # Affichage de la surface sphérique
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='brown', alpha=0.6, label='Astéroïde')

        for i, orb in enumerate(self.orbites):
            roe = [orb.x_r, orb.y_r, orb.a_r, orb.E_r, orb.A_z, orb.gamma]

            traj = []

            # Calculer la trajectoire
            for theta_deg in np.linspace(0, 360, num_points):
                roe_step = roe.copy()
                roe_step[3] = theta_deg
                pos, _ = roe2state(roe_step, sat_n, z_roe1='A_z', z_roe2='gamma')
                traj.append(pos)

            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"Orbit {i+1} (γ={orb.gamma:.1f}°, A_z={orb.A_z:.0f}m)")

            # Affichage des points de référence
            if show_reference_points:
                ref_points = orb.get_reference_points(sat_n)
                ref_points = np.array(ref_points)
                #print(ref_points)
                ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2],
                        c='red', s=10, label=f'Points Orbite {i+1}' if i == 0 else None)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        plt.tight_layout()
        plt.show()

    
    def init_plot(self, sat_n, num_points=500, show_reference_points=True):

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Orbites générées autour de l’astéroïde (via ROEs)")

        for i, orb in enumerate(self.orbites):
            roe = [orb.x_r, orb.y_r, orb.a_r, orb.E_r, orb.A_z, orb.gamma]
            traj = []
            for theta_deg in np.linspace(0, 360, num_points):
                roe_step = roe.copy()
                roe_step[3] = theta_deg
                pos, _ = roe2state(roe_step, sat_n, z_roe1='A_z', z_roe2='gamma')
                traj.append(pos)
            traj = np.array(traj)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"Orbit {i+1} (γ={orb.gamma:.1f}°, A_z={orb.A_z:.0f}m)")
            if show_reference_points:
                ref_points = orb.get_reference_points(sat_n)
                ref_points = np.array(ref_points)
                self.ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2],
                                c='red', s=10, label=f'Points Orbite {i+1}' if i == 0 else None)

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.legend()
        plt.tight_layout()

        # Créer un scatter plot pour la position satellite, initialement à (0,0,0)
        self.sat_point = self.ax.scatter([0], [0], [0], c='green', s=100, label='Satellite')
        plt.ion()  # mode interactif activé
        plt.show()


        



if __name__ == "__main__":
    factory = OrbiteFactory()
    sat_n = 0.001  # rad/s, exemple
    factory.plot_summary_orbits(sat_n=sat_n)

    for i, orb in enumerate(factory.orbites):
        print(f"Orbit {i+1}: {orb.get_label()}")


