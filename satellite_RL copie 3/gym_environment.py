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

sat_n = 5e-4


factory = OrbiteFactory()
orbites_dict = {i: orb for i, orb in enumerate(factory.orbites)}

reference_points = {}

for i in orbites_dict: 
    reference_points[i] = get_reference_ROE(orbites_dict[i])


parts_visited_on_orbit = initialize_parts_visited(reference_points)


### ===== REWARDS DEFINITIONS ===== ###

reward_buffer_full = 0
reward_delta_v = -5
reward_battery_discharged = -5
reward_recharge_battery = -1
reward_finished = 100
reward_dont_recharge_when_buffer_full = -1

### =============================== ###


### PARAMETERS DEFINITIONS ###

point_random = False # Fixed or random points
nombre_choix_angle_sur_orbite = 5 # Number of anomalies possible per orbit
if_noise = True # add some noise on values (batterie et data decay)
APF_true_biimpulsive_false = True ### Want to use APF or bi impulsive manoeuvers : APF = True, Biimpulsive = False

### ===================== ###



class DefEnvironnement(gym.Env):
    def __init__(self):

        # To get the orbit dictionnary
        self.orbites = orbites_dict  # dictionnaire {id: OrbiteROE}


        # To define the parameters of the simulation
        self.tau = 250
        self.sat_n = sat_n
        self.recharge_time = 100 # seconds
        self.data_transfer_time = 100  # seconds
        self.max_last_step = 300 #For 5 orbites
        #self.max_last_step = 600 #For 10 orbites


        # Initialisation of parameters
        self.parts_visited_on_orbit = parts_visited_on_orbit
        self.last_delta_v_total = 0.0 # We let the last_delta_total here even if it's not used just in case
        self.total_time = 0
        self.current_theta_deg = 0
        self.verification = []
        self.deltaV_tab = []
        self.current_step = 0 
        self.nombre_transferts = 0
        self.cout_total_transfert = 0
        self.last_episode_transferts = 0
        self.last_episode_cout = 0
        self.last_max_buffer_state = 0
        self.last_min_battery_state = 100
        self.real_result_battery = 0
        self.real_result_buffer = 0



        ### Important parameters
        self.point_random = point_random
        self.noise = if_noise
        self.satellite = Satellite()
        self.data_level = self.satellite.data_level
        self.battery_level = self.satellite.battery_level
        self.num_orbits = len(self.orbites)


        # Observation : [current_orbit, current_theta_deg, current_battery, data, timestep_on_each_orbit]
        low_obs = np.array([0, -180, 0, 0] + [0]*len(self.orbites), dtype=np.float32)
        high_obs = np.array([self.num_orbits, 180, 100, 100] + [1]*len(self.orbites), dtype=np.float32)


        self.orbite_change = False
        self.observation_space = spaces.Box(low=low_obs, high=high_obs)
        self.action_space = spaces.Discrete(self.num_orbits * nombre_choix_angle_sur_orbite + 2)

        self.reset()



    def _get_obs(self):

        #Battery and buffer
        battery = self.battery_level / 100
        data = self.data_level / 100

        # Satellite position
        num_orbite = float(self.satellite.current_orbit)
        current_theta_deg = state2roe([self.satellite.position, self.satellite.velocity], self.sat_n)[3]

        #Timestep dictionnary
        timestep_on_each_orbit2 = self.satellite.time_step_on_each_orbit

        #Observations
        obs = np.concatenate([
            [num_orbite, current_theta_deg, battery, data],
            timestep_on_each_orbit2
        ])

        return obs.astype(np.float32)



    def reset(self, seed=None, options=None):

        self.current_step = 0
        self.parts_visited_on_orbit = initialize_parts_visited(reference_points)
        self.satellite = Satellite()
        self.satellite.total_time = 0
        self.nombre_transferts = 0
        self.cout_total_transfert = 0

        ### Random points or not ###

        if self.point_random :
            a = np.random.randint(0, 10)
            b = np.random.randint(0, self.num_orbits)
            self.satellite.current_orbit = b
            initial_roe_state = reference_points[b][a]
        
        else : 
            self.satellite.current_orbit = 0
            orbite_init = self.orbites[0]
            initial_roe_state = reference_points[0][5] #Pour les fichiers dqn_1pointdedepart_fini
        
        ### ===================== ###
        

        self.verification = np.zeros(self.num_orbits, dtype=float)
        self.orbite_change = False
        self.satellite.position, self.satellite.velocity = roe2state(initial_roe_state, self.sat_n)
        self.done = False
        self.satellite.state = np.array([self.satellite.position, self.satellite.velocity])
        self.current_theta_deg = initial_roe_state[3]

        self.battery_level = 100
        self.data_level = 0

        # For tensorboard #
        self.last_max_buffer_state = 0
        self.last_min_battery_state = 100
        ### ========== ###


        self.satellite.time_step_on_each_orbit = np.zeros(self.num_orbits, dtype=float)
        

        return self._get_obs(), {}


    def step(self, action):

        self.current_step += 1
        dt = self.tau
        reward = 0


        ### ACTIONS SUR SATELLITE EN COURS OU FIN ###

        if self.battery_level <= 20 : ### -30 battery under 20% and -3 for recharging
            reward += reward_battery_discharged

        if self.data_level == 0 and action == self.num_orbits * 4 + 1 : 
            reward += -1

        ### ===================================== ###


        ### ACTIONS QUE PEUT EFFECTUER LE SATELLITE ###

       # 3. Action : Orbit change
        if action < self.num_orbits * nombre_choix_angle_sur_orbite  :
            orbit_target = action // nombre_choix_angle_sur_orbite  #Target orbit
            theta_target_id = action % nombre_choix_angle_sur_orbite  # theta target orbit
            theta_target_deg = theta_target_id * 360 / nombre_choix_angle_sur_orbite - 180 # in degre


            if orbit_target != self.satellite.current_orbit:
                if self.battery_level - self.satellite.battery_capacity * 0.2 > 0:

                    if self.noise: 
                        self.battery_level -= self.satellite.battery_capacity * random.uniform(0.15, 0.2)

                    else: 
                        self.battery_level -= self.satellite.battery_capacity * 0.2

                    self.orbite_change = True

                    # Get target ROE
                    target_roe = self.orbites[orbit_target]

                    # target ROE with the right anomaly
                    roe_target_state = [target_roe.x_r, target_roe.y_r, target_roe.a_r, theta_target_deg, target_roe.A_z, target_roe.gamma]


                    ### Bi-impulsive or APF ###

                    if APF_true_biimpulsive_false ==True:
                        delta_v_total, self.total_time, self.deltaV_tab = self.satellite.update_orbite_change(orbite_target=roe_target_state, sat_n=self.sat_n, total_time=self.total_time, deltaV_tab=self.deltaV_tab)

                    else: 
                        delta_v_total = self.satellite.orbit_transfer_2(orbite_target=target_roe, sat_n=self.sat_n)

                    ### =================== ###


                    self.satellite.current_orbit = orbit_target
                    self.last_delta_v_total = np.linalg.norm(delta_v_total)


                    ### Delta-v reward ###

                    reward += reward_delta_v * self.last_delta_v_total

                    ### ============== ###


                    self.nombre_transferts += 1
                    self.cout_total_transfert += reward_delta_v * self.last_delta_v_total

                else:
                    # Not enough battery, so no transfer
                    self.last_delta_v_total = 0

            else:
                # on the same orbit so no change
                self.last_delta_v_total = 0  



        # 5. Action : recharge #

        if action == self.num_orbits * nombre_choix_angle_sur_orbite :
            self.battery_level += self.satellite.battery_capacity * dt / self.recharge_time

            reward += reward_recharge_battery

            # Battery full
            if self.battery_level >= self.satellite.battery_capacity:
                self.battery_level = self.satellite.battery_capacity

            self.last_delta_v_total = 0.0
        
        # =================== #


        # 6. Action : Data transfer #

        if action == self.num_orbits * nombre_choix_angle_sur_orbite + 1 :
            self.data_level -= self.satellite.data_capacity * dt / self.data_transfer_time

            # Noise
            if self.noise :
                self.battery_level -= self.satellite.battery_capacity * random.uniform(0.15, 0.25)
            else : 
                self.battery_level -= self.satellite.battery_capacity * 0.2 

            # If you want a buffer full reward
            if self.data_level > self.satellite.data_capacity * 0.9 : 
                reward += reward_buffer_full

            # Data buffer empty
            if self.data_level < 0 : 
                self.data_level = 0

            self.last_delta_v_total = 0.0

        # ======================= #


        ### Update time-step on orbits and assign reward for new timestep ###

        if action < self.num_orbits * nombre_choix_angle_sur_orbite:
            if self.data_level < self.satellite.data_capacity:

                    reward, self.data_level = self.satellite.update_time_step_on_orbit(self.satellite.current_orbit, self.num_orbits, dt, reward, self.data_level, reward_buffer_full, self.satellite.time_step_on_each_orbit)

                    ### POUR CHECKER L'AVANCEMENT DANS L'EXPLORATION DU SATELLITE ###

                    if reward > 0 :

                        marqueur = self.satellite.has_visited_a_part_of_orbit(self.satellite.current_orbit, self.sat_n, self.parts_visited_on_orbit, tolerance=10)
                        #print(marqueur, "\n")

                        marqueur1 = self.satellite.has_visited_whole_orbit(self.satellite.current_orbit, self.sat_n, self.parts_visited_on_orbit)
                        marqueur2 = self.satellite.has_already_visited_a_part_of_orbit(self.satellite.current_orbit, self.sat_n, self.parts_visited_on_orbit, tolerance=10.0)
        
        ### ============================================================ ###


        ### MISE À JOUR SATELLITE ###

        # Update satellite position
        if not self.orbite_change :
            self.satellite.update_position_velocity(self.orbites[self.satellite.current_orbit], sat_n = self.sat_n, tau = dt)
            self.total_time += dt
            self.deltaV_tab.append(0)


        self.orbite_change = False

        # New current theta_deg
        self.current_theta_deg = state2roe([self.satellite.position, self.satellite.velocity], sat_n)[3]

        

        # Noise
        if self.noise :
            self.battery_level -= self.satellite.battery_capacity * random.uniform(0.003, 0.007)

        else : 
            self.battery_level -= self.satellite.battery_capacity * 0.005

        self.satellite.total_time += self.tau

        done = False

        # Terminated
        if self.satellite.if_orbit_fully_explored(): 
            reward += reward_finished
            done = True
            print("A")
        
        # Forced to end
        if self.current_step > self.max_last_step : 
            done = True


        ### Tensorboard ###

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

        ### ========== ###

        return self._get_obs(), reward, done, False, {}

    def render(self):
        pass

    def close(self):
        pass



### To test the environment with random actions ###

"""env = DefEnvironnement()
obs, _ = env.reset()
done = False

fig, ax, point, scatter_points, battery_bar, data_bar, orbite_bars, time_text = fa.setup_plot(env)

action = 6

while not done:
    action = env.action_space.sample()  # agent random

    obs, reward, done, _, info = env.step(action)

    fa.update_plot(env, point, ax, scatter_points=scatter_points, battery_bar=battery_bar, data_bar=data_bar, orbit_bars=orbite_bars, time_text=time_text)
    print(action)
    print(reward)
    print(obs)"""

### =========================================== ###
