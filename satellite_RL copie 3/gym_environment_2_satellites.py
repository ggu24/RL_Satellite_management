import numpy as np
import gymnasium as gym
from gymnasium import spaces
from satellite import Satellite
from def_orbites import OrbiteFactory
from fonctions_orbites import initialize_parts_visited, get_reference_ROE
from roe_functions import roe2state
from roe_functions import state2roe
import fonctions_affichage as fa
import copy
import random
import time

sat_n = 5e-4


#### CODE POUR PLUSIEURS SATELLITES DIFFÉRENTS EN PRENANT LES MÊMES OBSERVATIONS
#
#
#

nombre_de_satellites = 4 ## ATTENTION : NOMBRE_DE_SATELLITES < SELF.NUM_ORBITS


factory = OrbiteFactory()
orbites_dict = {i: orb for i, orb in enumerate(factory.orbites)}

reference_points = {}

for i in orbites_dict: 
    reference_points[i] = get_reference_ROE(orbites_dict[i])


visited_flags = {}

for i, points in reference_points.items():
    visited_flags[i] = np.zeros(len(points), dtype=np.float32)



parts_visited_on_orbit = initialize_parts_visited(reference_points)


### DEFINITIONS DES REWARDS ###

reward_buffer_full = 0
reward_delta_v = -5
reward_battery_discharged = -5
reward_recharge_battery = -1
reward_finished = 100
reward_dont_recharge_when_buffer_full = -1


### DEFINITIONS DES PARAMÈTRES ###

nombre_choix_angle_sur_orbite = 5 # Pour savoir à combien d'endroits différents sur une orbite le satellite peut se transférer
if_noise = True # POur savoir si on veut rajouter du bruit sur certains paramètres (batterie et data decay)
action_recharge_and_transfer = False

class DefEnvironnement_n_satellites(gym.Env):
    def __init__(self):
        self.orbites = orbites_dict  # dictionnaire {id: OrbiteROE}
        self.tau = 250

        self.sat_n = sat_n
        self.parts_visited_on_orbit = parts_visited_on_orbit

        self.rewarded_orbits = set()

        # Paramètres liés au timing des opérations (en secondes)
        self.recharge_time = 100 # en secondes
        self.data_transfer_time = 100  # en secondes

        self.visited_flags = visited_flags

        self.last_delta_v_total = 0.0

        self.current_theta_deg = 0

        self.current_step = 0 

        ###FONCTIONS PLUSIEURS SATELLITES####

        self.nombre_satellites = nombre_de_satellites
        self.current_satellite = 0
        self.initial_position_satellite = {}

        self.nombre_transferts = 0
        self.cout_total_transfert = 0

        self.last_episode_transferts = 0
        self.last_episode_cout = 0
        self.last_max_buffer_state = 0
        self.last_min_battery_state = 100
        self.real_result_battery = 0
        self.real_result_buffer = 0

        self.noise = if_noise

        self.battery_on_satellites =  [None] * nombre_de_satellites
        self.data_on_satellites = [None] * nombre_de_satellites
        self.current_theta_deg_satellites = [None] * nombre_de_satellites
        self.current_ROE_satellites = [None] * nombre_de_satellites
        self.current_orbit_satellites = [None] * nombre_de_satellites

        self.max_last_step = 300 #Pour 5 orbites

        #self.max_last_step = 600 #Pour 10 orbites

        self.satellite = Satellite()

        self.data_level = self.satellite.data_level
        self.battery_level = self.satellite.battery_level

        self.previous_last_delta_v = [None] * nombre_de_satellites

        self.num_orbits = len(self.orbites)

        # Observation : [current_satellite, current_orbit, current_theta_deg, current_battery, data, last_dv, self.visited_flags]

        if action_recharge_and_transfer:
            low_obs = np.array([0, 0, -180, 0, 0] + [0]*len(self.orbites), dtype=np.float32)
            high_obs = np.array([self.nombre_satellites, self.num_orbits, 180, 100, 100] + [1]*len(self.orbites), dtype=np.float32)

        else : 
            low_obs = np.array([0, 0, -180] + [0]*len(self.orbites), dtype=np.float32)
            high_obs = np.array([self.nombre_satellites, self.num_orbits, 180] + [1]*len(self.orbites), dtype=np.float32)

        self.orbite_change = False

        self.observation_space = spaces.Box(low=low_obs, high=high_obs)

        if action_recharge_and_transfer:
            self.action_space = spaces.Discrete(self.num_orbits * nombre_choix_angle_sur_orbite + 2)
        
        else : 
            self.action_space = spaces.Discrete(self.num_orbits * nombre_choix_angle_sur_orbite)

        self.reset()



    def _get_obs(self):

        current_satellite = self.current_satellite
        battery = self.battery_level / 100
        data = self.data_level / 100

        num_orbite = float(self.satellite.current_orbit)

        current_theta_deg = state2roe([self.satellite.position, self.satellite.velocity], self.sat_n)[3]

        timestep_on_each_orbit2 = self.satellite.time_step_on_each_orbit

        if action_recharge_and_transfer:
            obs = np.concatenate([
                [current_satellite, num_orbite, current_theta_deg, battery, data],
                timestep_on_each_orbit2
            ])

        else : 
            obs = np.concatenate([
                [current_satellite, num_orbite, current_theta_deg],
                timestep_on_each_orbit2
            ])

        return obs.astype(np.float32)



    def reset(self, seed=None, options=None):

        #print("[DEBUG] reset() a été appelé")

        self.current_satellite = 0
        self.current_step = 0
        self.parts_visited_on_orbit = initialize_parts_visited(reference_points)

        self.satellite = Satellite()

        self.satellite.total_time = 0

        self.previous_last_delta_v = 0

        self.nombre_transferts = 0

        self.cout_total_transfert = 0
    
        for i in range(nombre_de_satellites):
            self.satellite.current_orbit = i
            orbite_init = self.orbites[i]
            initial_roe_state = reference_points[i][5] #Pour les fichiers dqn_1pointdedepart_fini
            self.initial_position_satellite[i] = reference_points[i][5]
            self.current_orbit_satellites[i] = self.satellite.current_orbit


        self.orbite_change = False

        self.satellite.position, self.satellite.velocity = roe2state(initial_roe_state, self.sat_n)
        self.done = False
        self.satellite.state = np.array([self.satellite.position, self.satellite.velocity])
        self.satellite.time_step_on_each_orbit = np.zeros(self.num_orbits, dtype=float)

        for i in range(self.nombre_satellites): 
            self.data_on_satellites[i] = 0
            self.battery_on_satellites[i] = 100
            self.current_theta_deg_satellites[i] = self.initial_position_satellite[i][3]

    
        self.battery_level = 100
        self.data_level = 0

        self.last_max_buffer_state = 0
        self.last_min_battery_state = 100

        self.current_theta_deg = initial_roe_state[3]

        return self._get_obs(), {}


    def step(self, action):


        #print(f"ACTION REÇUE: {action} — current orbit: {self.satellite.current_orbit}")

        self.current_satellite = self.current_step % self.nombre_satellites

        if self.current_step < self.nombre_satellites: 
            self.battery_level = 100
            self.data_level = 0
            self.satellite.position, self.satellite.velocity = roe2state(self.initial_position_satellite[self.current_satellite], self.sat_n)
            self.current_ROE_satellites[self.current_satellite] = self.initial_position_satellite[self.current_satellite]
            self.satellite.state = np.array([self.satellite.position, self.satellite.velocity])
            self.current_theta_deg = self.initial_position_satellite[self.current_satellite][3]
            self.current_orbit_satellites[self.current_satellite] = self.current_satellite
            self.satellite.current_orbit = self.current_orbit_satellites[self.current_satellite]

        else : 
            self.battery_level = self.battery_on_satellites[self.current_satellite]
            self.data_level = self.data_on_satellites[self.current_satellite]
            self.satellite.position, self.satellite.velocity = roe2state(self.current_ROE_satellites[self.current_satellite], self.sat_n)
            self.satellite.state = np.array([self.satellite.position, self.satellite.velocity])
            self.current_theta_deg = self.current_ROE_satellites[self.current_satellite][3]
            self.satellite.current_orbit = self.current_orbit_satellites[self.current_satellite]


        self.current_step += 1

        dt = self.tau

        ### ATTENTION NON PERMANENT : 
        #action = len(self.orbites)
        #########################

        if not action_recharge_and_transfer:
            self.battery_level = 100
            self.data_level = 0

        reward = 0

        ### ACTIONS SUR SATELLITE EN COURS OU FIN ###

        if self.battery_level <= 20 : ### -30 battery under 20% and -3 for recharging
            reward += reward_battery_discharged

        if self.data_level == 0 and action == self.num_orbits * 4 + 1 : 
            reward += -1

        ####  #####   ####

        #print(action)


        ### ACTIONS QUE PEUT EFFECTUER LE SATELLITE ###

       # 3. Action : changement d’orbite
        if action < self.num_orbits * nombre_choix_angle_sur_orbite  :
            orbit_target = action // nombre_choix_angle_sur_orbite 
            theta_target_id = action % nombre_choix_angle_sur_orbite 
            theta_target_deg = theta_target_id * 360 / nombre_choix_angle_sur_orbite - 180

            #print(theta_target_deg, "theta visé")

            if orbit_target != self.satellite.current_orbit:
                if self.battery_level - self.satellite.battery_capacity * 0.2 > 0:

                    if self.noise: 
                        self.battery_level -= self.satellite.battery_capacity * random.uniform(0.15, 0.2)

                    else: 
                        self.battery_level -= self.satellite.battery_capacity * 0.2

                    self.orbite_change = True

                    # Récupérer orbite cible
                    target_roe = self.orbites[orbit_target]

                    # On remplace theta par theta_target_deg dans roe
                    roe_target_state = [target_roe.x_r, target_roe.y_r, target_roe.a_r, theta_target_deg, target_roe.A_z, target_roe.gamma]

                    total_time = 0
                    total_deltav = []

                    delta_v_total, _, _ = self.satellite.update_orbite_change(roe_target_state, self.sat_n, total_time, total_deltav)

                    self.satellite.current_orbit = orbit_target
                    self.last_delta_v_total = np.linalg.norm(delta_v_total)

                    reward += reward_delta_v * self.last_delta_v_total


                    self.nombre_transferts += 1

                    self.cout_total_transfert += reward_delta_v * self.last_delta_v_total

                else:
                    # Batterie insuffisante, pas de changement
                    self.last_delta_v_total = 0

            else:
                self.last_delta_v_total = 0  # pas de dépense delta_v pour changer theta seul



        # 5. Action spéciale : recharge
        if action == self.num_orbits * nombre_choix_angle_sur_orbite :
            self.battery_level += self.satellite.battery_capacity * dt / self.recharge_time

            reward += reward_recharge_battery

            if self.battery_level >= self.satellite.battery_capacity:
                self.battery_level = self.satellite.battery_capacity
            
                #print("Batterie déjà pleine\n")

            self.last_delta_v_total = 0.0


        # 6. Action spéciale : transfer data
        if action == self.num_orbits * nombre_choix_angle_sur_orbite + 1 :
            self.data_level -= self.satellite.data_capacity * dt / self.data_transfer_time

            if self.noise :
                self.battery_level -= self.satellite.battery_capacity * random.uniform(0.15, 0.25)

            else : 
                self.battery_level -= self.satellite.battery_capacity * 0.2 

            if self.data_level > self.satellite.data_capacity * 0.9 : 
                reward += reward_buffer_full

            if self.data_level < 0 : 
                self.data_level = 0

                #print("Batterie déjà pleine\n")

            self.last_delta_v_total = 0.0



        if action < self.num_orbits * nombre_choix_angle_sur_orbite:
            if self.data_level < self.satellite.data_capacity:

                    reward, self.data_level = self.satellite.update_time_step_on_orbit(self.satellite.current_orbit, self.num_orbits, dt, reward, self.data_level, reward_buffer_full, self.satellite.time_step_on_each_orbit)


                    ### POUR CHECKER L'AVANCEMENT DANS L'EXPLORATION DU SATELLITE ###

                    if reward > 0 :

                        marqueur = self.satellite.has_visited_a_part_of_orbit(self.satellite.current_orbit, self.sat_n, self.parts_visited_on_orbit, tolerance=10)
                        #print(marqueur, "\n")

                        marqueur1 = self.satellite.has_visited_whole_orbit(self.satellite.current_orbit, self.sat_n, self.parts_visited_on_orbit)
                        marqueur2 = self.satellite.has_already_visited_a_part_of_orbit(self.satellite.current_orbit, self.sat_n, self.parts_visited_on_orbit, tolerance=10.0)



        ### MISE À JOUR SATELLITE ###

        if not self.orbite_change :
            self.satellite.update_position_velocity(self.orbites[self.satellite.current_orbit], sat_n = self.sat_n, tau = dt)

        self.orbite_change = False

        self.current_theta_deg = state2roe([self.satellite.position, self.satellite.velocity], sat_n)[3]

        done = False



        if self.noise :
            self.battery_level -= self.satellite.battery_capacity * random.uniform(0.003, 0.007)

        else : 
            self.battery_level -= self.satellite.battery_capacity * 0.005

        self.satellite.total_time += self.tau

        if self.satellite.if_orbit_fully_explored(): ###terminated
            reward += reward_finished
            done = True
            print("A")

        if self.current_step > self.max_last_step : 
            done = True

        if self.last_min_battery_state > self.battery_level : 
            self.last_min_battery_state = self.battery_level
        
        if self.last_max_buffer_state < self.data_level: 
            self.last_max_buffer_state = self.data_level

        if done == True : 
            self.last_episode_transferts = self.nombre_transferts
            self.last_episode_cout = self.cout_total_transfert

        if done == True : 
            self.real_result_buffer = self.last_max_buffer_state
            self.real_result_battery = self.last_min_battery_state

        #print(done)

        ###RÉCUPÉRATION DES DONNÉES IMPORTANTES SUR CHAQUE SATELLITE
        self.battery_on_satellites[self.current_satellite] = self.battery_level 
        self.data_on_satellites[self.current_satellite] = self.data_level 
        self.current_ROE_satellites[self.current_satellite] = state2roe([self.satellite.position, self.satellite.velocity], self.sat_n)
        self.current_ROE_satellites[self.current_satellite][3] = self.current_theta_deg
        self.current_orbit_satellites[self.current_satellite] = self.satellite.current_orbit

        return self._get_obs(), reward, done, False, {}


    def render(self):
        pass

    def close(self):
        pass


"""env = DefEnvironnement_n_satellites()
obs, _ = env.reset()
done = False

#fig, ax, point, scatter_points, battery_bar, data_bar = fa.setup_plot(env)
fig, ax, point, scatter_points, battery_bar, data_bar, orbite_bars = fa.setup_plot(env)

while not done:
    action = env.action_space.sample()  # agent random

    obs, reward, done, _, info = env.step(action)

    fa.update_plot(env, point, ax, scatter_points=scatter_points, battery_bar=battery_bar, data_bar=data_bar, orbit_bars=orbite_bars)
    time.sleep(0.01)
    print(action)
    print(reward)
    print(obs)"""
