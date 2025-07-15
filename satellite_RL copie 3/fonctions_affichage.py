import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from roe_functions import roe2state
from matplotlib import gridspec
import numpy as np

"""def setup_plot(env):
    plt.ion()  # mode interactif ON
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Satellite live tracking")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    scatter_points=[]

    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[4, 1])

    # Tracer les orbites (optionnel)
    for orb in env.orbites.values():
        roe = [orb.x_r, orb.y_r, orb.a_r, orb.E_r, orb.A_z, orb.gamma]
        traj = []
        for theta_deg in np.linspace(0, 360, 500):
            roe_step = roe.copy()
            roe_step[3] = theta_deg
            pos, _ = roe2state(roe_step, env.sat_n, z_roe1='A_z', z_roe2='gamma')
            traj.append(pos)
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.3)

    # Un point pour le satellite
    sat_pos = env.satellite.position
    point, = ax.plot(
    [sat_pos[0]], [sat_pos[1]], [sat_pos[2]],
    marker='o',          # forme du point
    color='blue',        # couleur
    markersize=10,       # taille (par défaut 6)
    label="Satellite")

    # Les points pour le passage du satellite à certains endroits :

    for orbit in env.parts_visited_on_orbit.values():
        for visited, roe_point in orbit:
            pos = np.array(roe2state(roe_point, env.sat_n))
            color = 'green' if visited else 'red'
            scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, s=10)
            scatter_points.append(scatter)  # on les garde
    
    #print(ref_points)

    # === Axes pour les barres de battery et data
    axbar = fig.add_subplot(spec[1])
    axbar.set_xlim(0, 1)
    axbar.set_ylim(0, 1)
    axbar.axis('off')

    battery_bar = axbar.barh(0.75, env.battery_level / 100, height=0.2, color='orange')[0]
    data_bar = axbar.barh(0.25, env.data_level / 100, height=0.2, color='cyan')[0]

    axbar.text(0, 0.75, 'Battery', va='center', ha='left', fontsize=9)
    axbar.text(0, 0.25, 'Data', va='center', ha='left', fontsize=9)
    

    ax.legend()
    plt.draw()
    plt.pause(0.001)
    return fig, ax, point, scatter_points, battery_bar, data_bar


def update_plot(env, point, ax, scatter_points, battery_bar, data_bar):
    # Mettre à jour la position du satellite
    sat_pos = env.satellite.position
    point.set_data([sat_pos[0]], [sat_pos[1]])
    point.set_3d_properties([sat_pos[2]])

    # Supprimer tous les anciens scatter points sauf le satellite
    # On garde l'objet "point", les autres (scatter) on les efface
    for scatter in scatter_points:
        scatter.remove()
    scatter_points.clear()

    # Ajouter à nouveau les points visités / non visités
    for orbit in env.parts_visited_on_orbit.values():
        for visited, roe_point in orbit:
            pos = roe2state(roe_point, env.sat_n)
            pos = np.array(pos)
            color = 'green' if visited else 'red'
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, s=10)

    battery_level = env.battery_level / 100
    data_level = env.data_level / 100

    # Mise à jour des barres
    battery_bar.set_width(battery_level)
    data_bar.set_width(data_level)

    plt.draw()
    plt.pause(0.001)"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import numpy as np
from roe_functions import roe2state

def setup_plot(env):
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    
    # Grille : [Orbites | Graphe 3D | Batterie/Data]
    spec = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1, 4, 1])
    
    # === Axes de progression des orbites ===
    orbit_progress = env.satellite.time_step_on_each_orbit
    num_orbits = len(orbit_progress)

    ax_orbits = fig.add_subplot(spec[0])
    ax_orbits.set_xlim(0, 1)
    ax_orbits.set_ylim(0, num_orbits)
    ax_orbits.invert_yaxis()
    ax_orbits.set_title("Progression orbites")
    ax_orbits.set_xticks([])
    ax_orbits.set_yticks(range(num_orbits))
    ax_orbits.set_yticklabels([f"Orbite {i+1}" for i in range(num_orbits)], fontsize=8)

    orbit_bars = []
    for i, value in enumerate(orbit_progress):
        bar = ax_orbits.barh(i, value, height=0.6, color='purple')[0]
        orbit_bars.append(bar)

    # === Axes du graphe 3D ===
    ax = fig.add_subplot(spec[1], projection='3d')
    ax.set_title("Satellite live tracking")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scatter_points = []
    for orb in env.orbites.values():
        roe = [orb.x_r, orb.y_r, orb.a_r, orb.E_r, orb.A_z, orb.gamma]
        traj = []
        for theta_deg in np.linspace(0, 360, 500):
            roe_step = roe.copy()
            roe_step[3] = theta_deg
            pos, _ = roe2state(roe_step, env.sat_n, z_roe1='A_z', z_roe2='gamma')
            traj.append(pos)
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.3)

    sat_pos = env.satellite.position
    point, = ax.plot([sat_pos[0]], [sat_pos[1]], [sat_pos[2]],
                     marker='o', color='blue', markersize=10, label="Satellite")

    for orbit in env.parts_visited_on_orbit.values():
        for visited, roe_point in orbit:
            pos = np.array(roe2state(roe_point, env.sat_n))
            color = 'green' if visited else 'red'
            scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, s=10)
            scatter_points.append(scatter)

    ax.legend()

    # === Axes pour Battery et Data ===
    axbar = fig.add_subplot(spec[2])
    axbar.set_xlim(0, 1)
    axbar.set_ylim(0, 1)
    axbar.axis('off')

    battery_bar = axbar.barh(0.75, env.battery_level / 100, height=0.2, color='orange')[0]
    data_bar = axbar.barh(0.25, env.data_level / 100, height=0.2, color='cyan')[0]

    axbar.text(0, 0.75, 'Battery', va='center', ha='left', fontsize=9)
    axbar.text(0, 0.25, 'Data', va='center', ha='left', fontsize=9)

    time_text = fig.text(0.95, 0.95, f"Total time: {env.total_time//60:.1f} min", ha='right', fontsize=10)

    plt.draw()
    plt.pause(0.001)

    return fig, ax, point, scatter_points, battery_bar, data_bar, orbit_bars, time_text


def update_plot(env, point, ax, scatter_points, battery_bar, data_bar, orbit_bars, time_text, show_points = False):
    # Mise à jour de la position du satellite
    sat_pos = env.satellite.position
    point.set_data([sat_pos[0]], [sat_pos[1]])
    point.set_3d_properties([sat_pos[2]])

    # Suppression des anciens points visités
    for scatter in scatter_points:
        scatter.remove()
    scatter_points.clear()

    # Ajout des points visités/non visités
    for orbit in env.parts_visited_on_orbit.values():
        for visited, roe_point in orbit:
            pos = np.array(roe2state(roe_point, env.sat_n))
            color = 'green' if visited else 'red'
            if show_points:
                scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, s=10) 
                scatter_points.append(scatter)

    # Mise à jour des barres de batterie et data
    battery_bar.set_width(env.battery_level / 100)
    data_bar.set_width(env.data_level / 100)

    # Mise à jour des barres de progression d'orbite
    for i, progress in enumerate(env.satellite.time_step_on_each_orbit):
        orbit_bars[i].set_width(progress)
    
    time_text.set_text(f"Total time: {env.total_time//60:.1f} min")

    plt.draw()
    plt.pause(0.001)
