import numpy as np
import matplotlib as plt
from gymnasium import spaces
import cv2
from def_orbites import OrbiteFactory
from satellite import Satellite
from scipy.constants import G
from fonctions_orbites import initialize_parts_visited, get_reference_ROE
from roe_functions import roe2state
import fonctions_affichage as fa


sat_n = 5e-4

print(sat_n)

factory = OrbiteFactory()
orbites_dict = {i: orb for i, orb in enumerate(factory.orbites)}

reference_points = {}

for i in orbites_dict: 
    reference_points[i] = get_reference_ROE(orbites_dict[i])


parts_visited_on_orbit = initialize_parts_visited(reference_points)


### DEFINITIONS DES REWARDS ###

reward_change_orbit_success = 100
reward_buffer_full = -1
reward_no_more_battery = -1
reward_both = reward_no_more_battery + reward_buffer_full
reward_orbit_already_visited = -10
reward_visited_whole_orbit = 5
reward_delta_v = -10
reward_new_point = 2
reward_point_already_visited = -10


### CRÉATION DE LA CLASSE

class Def_environnement:
    def __init__(self):
        self.orbites = orbites_dict  # dictionnaire {id: OrbiteROE}
        self.tau = 200

        self.sat_n = sat_n
        self.parts_visited_on_orbit = parts_visited_on_orbit

        # Paramètres liés au timing des opérations (en secondes)
        self.recharge_time = 20 * 60  # 20 min en secondes
        self.data_transfer_time = 10 * 60  # 10 min en secondes

        self.last_delta_v_total = 0.0

        self.rewarded_orbits = set()

        self.current_step = 0 
        self.max_last_step = 1000

        self.satellite = Satellite()

        self.data_level = self.satellite.data_level
        self.battery_level = self.satellite.battery_level

        self.num_orbits = len(self.orbites)


        # Observation : [pos (3), vel (3), battery, data, is_recharging]
        low_obs = np.array([-1e5]*3 + [-1e3]*3 + [0.0, 0.0] + [0, 0] + [0], dtype=np.float32)
        high_obs = np.array([1e5]*3 + [1e3]*3 + [self.battery_level, self.data_level] + [1, 1] + [5e3], dtype=np.float32)

        self.observation_space = spaces.Box(low=low_obs, high=high_obs)

        self.action_space = spaces.Discrete(len(self.orbites) + 3)  # actions = orbites cibles + 3 autres possibilités 

        self.reset()


    def _get_obs(self):
        pos = self.satellite.position if self.satellite.position is not None else np.zeros(3)
        vel = self.satellite.velocity if self.satellite.velocity is not None else np.zeros(3)
        battery = self.battery_level
        data = self.data_level

        flags = [
            float(self.satellite.is_recharging),
            float(self.satellite.is_transferring_data)
        ]

        dv = float(self.last_delta_v_total)

        #print("Types and shapes:")
        #print("pos", type(pos), np.shape(pos))
        #print("vel", type(vel), np.shape(vel))
        #print("battery", type(battery), battery)
        #print("data", type(data), data)
        #print("flags", type(flags), np.shape(flags))
        #print("dv", type(dv), dv)

        return np.concatenate([pos, vel, [battery, data], flags, [dv]]).astype(np.float32)



    def reset(self, seed=None, options=None):

        self.current_step = 0

        self.satellite = Satellite()
        self.satellite.current_orbit = 0
        orbite_init = self.orbites[0]
        initial_roe_state = reference_points[0][0]
        self.satellite.position, self.satellite.velocity = roe2state(initial_roe_state, self.sat_n)
        self.done = False
        self.satellite.state = np.array([self.satellite.position, self.satellite.velocity])
        self.battery_level = 100
        self.data_level = 100
        return self._get_obs(), {}


    def step(self, action):

        print(f"ACTION REÇUE: {action} — current orbit: {self.satellite.current_orbit}")

        self.current_step += 1

        dt = self.tau

        ### ATTENTION NON PERMANENT : 
        #action = len(self.orbites)
        #########################

        reward = 0


        ### ACTIONS SUR SATELLITE EN COURS OU FIN ###

        if self.satellite.is_recharging:
            if self.battery_level >= self.satellite.battery_capacity:
                self.battery_level = self.satellite.battery_capacity
                self.satellite.is_recharging = False
            
            self.battery_level += self.satellite.battery_capacity * dt / self.recharge_time

            if self.battery_level > self.satellite.battery_capacity : 
                self.battery_level = self.satellite.battery_capacity

            self.satellite.update_position_velocity(self.orbites[self.satellite.current_orbit], self.sat_n, dt)
            print("Ne fais rien car se recharge\n")
            return self._get_obs(), reward, False, False, {}

        if self.satellite.is_transferring_data:
            if self.data_level >= self.satellite.data_capacity:
                self.data_level = self.satellite.data_capacity
                self.satellite.is_transferring_data = False

            self.data_level += self.satellite.data_capacity * dt / self.data_transfer_time
            self.battery_level -= self.satellite.battery_capacity * 0.2 * dt / self.recharge_time

            if self.data_level > self.satellite.data_capacity : 
                self.data_level = self.satellite.data_capacity
            
            self.satellite.update_position_velocity(self.orbites[self.satellite.current_orbit], self.sat_n, dt)
            print("Ne fais rien car transfère de la DATA\n")
            return self._get_obs(), reward, False, False, {}


        #### CAS DE BATTERIE VIDE ET / OU BUFFER PLEIN ####

        # Cas critique : batterie vide et buffer plein
        if self.battery_level <= 0 and self.data_level <= 0 :
            self.satellite.is_recharging = True
            self.satellite.update_position_velocity(self.orbites[self.satellite.current_orbit], self.sat_n, dt)
            return self._get_obs(), reward_both, False, False, {"forced_action": "recharge_due_to_battery_and_buffer"}

        # Batterie vide uniquement
        elif self.battery_level <= 0:
            self.satellite.is_recharging = True
            self.satellite.update_position_velocity(self.orbites[self.satellite.current_orbit], self.sat_n, dt)
            return self._get_obs(), reward_no_more_battery, False, False, {"forced_action": "recharge_due_to_battery"}

        # Buffer plein uniquement
        elif self.data_level <= 0 : 
            self.satellite.is_transferring_data = True
            self.satellite.update_position_velocity(self.orbites[self.satellite.current_orbit], self.sat_n, dt)
            return self._get_obs(), reward_buffer_full, False, False, {"forced_action": "transfer_due_to_buffer"}
        


        ### POUR CHECKER L'AVANCEMENT DANS L'EXPLORATION DU SATELLITE ###

        marqueur = self.satellite.has_visited_a_part_of_orbit(self.satellite.current_orbit, self.sat_n, self.parts_visited_on_orbit, tolerance=10.0)
        print(marqueur, "\n")
        if marqueur == True :
            self.data_level -= self.satellite.data_capacity * 0.1
            reward += reward_new_point

        if self.satellite.has_visited_whole_orbit(self.satellite.current_orbit, self.sat_n, self.parts_visited_on_orbit):
            if self.satellite.current_orbit not in self.rewarded_orbits:
                reward += reward_visited_whole_orbit
                self.rewarded_orbits.add(self.satellite.current_orbit)


        marqueur2 = self.satellite.has_already_visited_a_part_of_orbit(self.satellite.current_orbit, self.sat_n, self.parts_visited_on_orbit, tolerance=10.0)

        if marqueur2 == True : 
            reward += reward_point_already_visited

        ####  #####   ####

        #print(action)


        ### ACTIONS QUE PEUT EFFECTUER LE SATELLITE ###

        # 3. Action : changement d’orbite
        if isinstance(action, int) and action < self.num_orbits :
            if action != self.satellite.current_orbit:

                print("Changement d'orbite\n")

                target_roe = self.orbites[action]

                delta_v_total = self.satellite.update_orbite_change(
                    orbite_target=[target_roe.x_r, target_roe.y_r, target_roe.a_r, None, target_roe.A_z, target_roe.gamma], sat_n = self.sat_n)
                
                self.satellite.current_orbit = action

                #print(self.satellite.current_orbit, ":   2")
                
                if self.satellite.has_visited_whole_orbit(action, self.sat_n, self.parts_visited_on_orbit):
                    reward += reward_orbit_already_visited
                
                self.last_delta_v_total = np.linalg.norm(delta_v_total) # Met en observation le dernier delta_v, sinon ne met rien 

                #print(self.last_delta_v_total, "le dernier delta v \n")

                reward += reward_change_orbit_success
                reward += reward_delta_v * self.last_delta_v_total
                print(reward_delta_v * self.last_delta_v_total / 0.01, "reward changement orbite delta v\n")

                
            else : 
                self.last_delta_v_total = 0
                print("Est déjà sur le même orbite\n")


        # 4. Action spéciale : idle
        elif action == self.num_orbits:
            self.satellite.update_position_velocity(self.orbites[self.satellite.current_orbit], sat_n = self.sat_n, tau = dt)
            self.satellite.position = self.satellite.state[0]
            self.satellite.position = self.satellite.state[1]
            print("Reste sans rien faire\n")

            self.last_delta_v_total = 0.0


        # 5. Action spéciale : recharge
        elif action == self.num_orbits + 1:

            self.satellite.is_recharging = True

            if self.battery_level >= self.satellite.battery_capacity: 
                self.satellite.is_recharging = False 
                print("Batterie déjà pleine\n")

            if self.satellite.is_recharging:
                print("Reste sur son orbite mais recharge\n")

            self.last_delta_v_total = 0.0


        # 6. Action spéciale : transfer data
        elif action == self.num_orbits + 2:
            self.satellite.is_transferring_data = True

            if self.data_level >= self.satellite.data_capacity: 
                self.satellite.is_transferring_data = False 
                print("Batterie déjà pleine\n")

            if self.satellite.is_transferring_data:
                print("Reste sur son orbite mais transfère de la DATA\n")

            self.last_delta_v_total = 0.0


        ### MISE À JOUR SATELLITE ###



        self.satellite.update_position_velocity(self.orbites[self.satellite.current_orbit], sat_n = self.sat_n, tau = dt)


        
        done = self.current_step >= self.max_last_step

        #print(self.satellite.current_orbit, "Orbite actuelle")
        #print(self.parts_visited_on_orbit[0])

        self.battery_level -= self.satellite.battery_capacity * 0.005
        #self.satellite.data_level -= self.satellite.data_capacity * 0.005

        print(self.last_delta_v_total, "delta_V_total")



        return self._get_obs(), reward, done, False, {}




env = Def_environnement()
obs, _ = env.reset()
done = False

fig, ax, point, scatter_points, battery_bar, data_bar = fa.setup_plot(env)

while not done:
    action = np.random.choice(env.action_space.n)  # agent random
    obs, reward, done, _, info = env.step(action)
    fa.update_plot(env, point, ax, scatter_points=scatter_points, battery_bar = battery_bar, data_bar = data_bar)
    #print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")
    print(f"Observation: {obs}")

