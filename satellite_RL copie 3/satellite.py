import numpy as np
from hcw_propagator import propagate
from scipy.constants import G
from control_functions import apf_renevey
from roe_functions import roe_difference, state2roe, roe2state
from bi_impulsive_function_2 import bi_impulsive_transfer

class Satellite:

    def __init__(self, mass=500, battery_capacity_wh=100, data_capacity_total=100):

        # Propriétés physiques

        self.mass = mass  # kg
        self.battery_capacity = battery_capacity_wh  # capacité max de la batterie
        self.battery_level = battery_capacity_wh  # batterie pleine au départ

        # Data buffer (en km parcourus non explorés)
        self.data_capacity = data_capacity_total
        self.data_level = 0  # points parcourus en collecte (initialement vide)

        #Conditions particulières
        self.is_recharging = False
        self.is_transferring_data = False

        # Stockage des orbites visitées sous forme de dictionnaire
        self.visited_orbits = {}
        self.parts_visited_by_satellite_on_orbit = {}

        self.total_time = 0

        # État du satellite 
        self.state = None

        self.stop = 0


        # Orbite actuelle 
        self.current_orbit = None
        self.count = 0


        # Paramètres liés au timing des opérations (en secondes)
        self.recharge_time = 20 * 60  # 20 min en secondes
        self.data_transfer_time = 10 * 60  # 10 min en secondes


        # Timers internes pour gérer durée recharge/data transfer
        self._recharge_timer = 0
        self._data_transfer_timer = 0

        self.time_step_on_each_orbit = []



    def start_recharge(self):
        self.is_recharging = True
        self._recharge_timer = self.recharge_time



    def start_data_transfer(self):
        if self.data_level > 0:
            self.is_transferring_data = True
            self._data_transfer_timer = self.data_transfer_time
        else:
            print("Impossible de transférer data : buffer vide")



    """def update(self, dt):
        
        #Met à jour l'état du satellite au niveau des évenements particuliers à chaque pas de temps dt (secondes)
        # Recharge
        if self.is_recharging:
            recharge_rate = self.battery_capacity / self.recharge_time  # Wh/s
            self.battery_level = min(self.battery_capacity, self.battery_level + recharge_rate * dt)
            self._recharge_timer -= dt
            if self._recharge_timer <= 0 or self.battery_level >= self.battery_capacity:
                self.is_recharging = False
                self.battery_level = self.battery_capacity

        # Transfert de données
        if self.is_transferring_data:
            battery_consump_rate = 0.3 * self.battery_capacity / self.data_transfer_time  # Wh/s
            self.battery_level = max(0, self.battery_level - battery_consump_rate * dt)
            self._data_transfer_timer -= dt
            if self._data_transfer_timer <= 0 or self.data_level <= 0:
                self.is_transferring_data = False
                self.data_level = 0

            # Changement d'orbite
        if self.is_changing_orbit:
            self._orbit_change_timer -= dt
            if self._orbit_change_timer <= 0:
                self.is_changing_orbit = False
                self._delta_v = 0
                self._orbit_change_duration = 0
                self._orbit_change_timer = 0
                # Ici tu peux mettre à jour current_orbit""" # N'est plus utile car on considère que le satellite se téléporte"""

    def has_visited_a_part_of_orbit_2(self, orbit_id, sat_n, parts_visited_on_orbit, data_level, tolerance=0.05):
        """
        Vérifie si l’état actuel du satellite est suffisamment proche d’un point de référence sur l'orbite donné et le marque si il est assez proche

        :param orbit_id: ID de l'orbite actuelle
        :param sat_n: valeur de n pour l'orbite de référence
        :param parts_visited_on_orbit: dict {orbit_id: [(bool, roe_point), ...]}
        :param tolerance: seuil de distance pour considérer qu’un point est visité (en norme ROE)
        """

        marqueur = False

        # Convertir l'état actuel en ROE
        actual_roe = state2roe(self.state, sat_n)

        for i, (visited, roe_point) in enumerate(parts_visited_on_orbit[orbit_id]):
            if not visited:
                dist = roe_difference(roe_point, actual_roe)
                if dist < tolerance:
                    # Marquer comme visité
                    parts_visited_on_orbit[orbit_id][i] = (True, roe_point)
                    marqueur = True
                    data_level += 0.1 * self.data_capacity

        return marqueur



    

    def has_visited_a_part_of_orbit(self, orbit_id, sat_n, parts_visited_on_orbit, tolerance=0.05):
        """
        Vérifie si l’état actuel du satellite est suffisamment proche d’un point de référence sur l'orbite donné et le marque si il est assez proche

        :param orbit_id: ID de l'orbite actuelle
        :param sat_n: valeur de n pour l'orbite de référence
        :param parts_visited_on_orbit: dict {orbit_id: [(bool, roe_point), ...]}
        :param tolerance: seuil de distance pour considérer qu’un point est visité (en norme ROE)
        """

        marqueur = False

        # Convertir l'état actuel en ROE
        actual_roe = state2roe(self.state, sat_n)

        for i, (visited, roe_point) in enumerate(parts_visited_on_orbit[orbit_id]):
            if not visited:
                dist = roe_difference(roe_point, actual_roe)
                if dist < tolerance:
                    # Marquer comme visité
                    parts_visited_on_orbit[orbit_id][i] = (True, roe_point)
                    marqueur = True

        return marqueur
    
    def count_validated_points(self, parts_visited_on_orbits):
        """
        Compte le nombre total de points validés (True) dans le dictionnaire.

        :param parts_visited_on_orbits: dict {orbit_id: [(bool, roe_point), ...]}
        :return: int, nombre total de points validés
        """
        total_validated = 0
        for orbit_id, points in parts_visited_on_orbits.items():
            total_validated += sum(1 for visited, _ in points if visited)
        return total_validated
    

    def has_already_visited_a_part_of_orbit(self, orbit_id, sat_n, parts_visited_on_orbit, tolerance=0.05):
        """
        Vérifie si l’état actuel du satellite est suffisamment proche d’un point de référence sur l'orbite donné et le marque si il est assez proche

        :param orbit_id: ID de l'orbite actuelle
        :param sat_n: valeur de n pour l'orbite de référence
        :param parts_visited_on_orbit: dict {orbit_id: [(bool, roe_point), ...]}
        :param tolerance: seuil de distance pour considérer qu’un point est visité (en norme ROE)
        """

        marqueur = False

        # Convertir l'état actuel en ROE
        actual_roe = state2roe(self.state, sat_n)

        for i, (visited, roe_point) in enumerate(parts_visited_on_orbit[orbit_id]):
            dist = roe_difference(roe_point, actual_roe)
            if dist < tolerance:
                if visited:
                    marqueur = True

        return marqueur



    def has_visited_whole_orbit(self, orbit_id, sat_n, parts_visited_on_orbit):
        """
        Vérifie si tous les points de référence d'une orbite ont été visités.

        :param orbit_id: ID de l'orbite à vérifier
        :param sat_n: (non utilisé ici mais laissé pour cohérence)
        :param parts_visited_on_orbit: dict {orbit_id: [(bool, roe_point), ...]}
        :return: True si tous les points de l'orbite ont été visités, False sinon
        """

        if orbit_id not in parts_visited_on_orbit:
            return False

        return all(visited for visited, _ in parts_visited_on_orbit[orbit_id])



    def update_position_velocity(self, orbite, sat_n, tau):
        """
        Met à jour la position et la vitesse du satellite en utilisant les équations HCW
        si le satellite n'est pas en train de changer d'orbite.

        :param orbite: instance de Orbite (orbite actuelle du satellite)
        :param tau: durée du pas de temps (en secondes)
        """

        # État initial (position et vitesse)
        state0 = self.state
        t_span = (0, tau)
        t_eval = [0, tau]  # évaluer à t = tau


        _, state = propagate(state0, t_span, sat_n, lin_ctrl=None, t_eval = t_eval)


        # Mise à jour du satellite

        self.state = state

        self.position = state[0]  # position à t = tau
        self.velocity = state[1] # vitesse à t = tau

        

    


    def update_orbite_change(self, orbite_target, sat_n, total_time, deltaV_tab): 
        """
        Met à jour la position et la vitesse du satellite en utilisant les équations HCW si le satellite est en train de changer d'orbite.

        :param orbite_target: objectif d'orbite
        :param sat_n: valeur n de notre satellite

        :return total_time: temps total passé à faire le transfert
        """

        tau_control = 120 # Toutes les 1 secondes

        tau_propagate = 120

        tau_span = (0, tau_propagate)

        t_eval = [0, tau_propagate]

        error_limit = 0.005

        time_transfer = 0

        delta_V_total = 0

        error = error_limit + 1


        #delta_V, error = apf_renevey(self.state, orbite_target, sat_n, tau_control, np.zeros(3))

        #delta_V_total += delta_V[0]

        #total_time += tau_control

        delta_V = [0, 0]

        while error > error_limit : 

            delta_V, error = apf_renevey(self.state, orbite_target, sat_n, tau_control, delta_V[1])

            self.position = self.state[0]
            self.velocity = self.state[1]

            self.velocity += delta_V[0]

            time_transfer += 120

            total_time += 120

            self.state = [self.position, self.velocity]

            _, states = propagate(self.state, tau_span, sat_n, lin_ctrl = None, t_eval = t_eval)

            self.state = [states[0], states[1]] 

            deltaV_tab.append(np.linalg.norm(delta_V[0]))

            #print(self.state, "self.state \n\n")
            #print(delta_V, "delta_V\n\n")

            #print(error, "\n")

            delta_V_total += np.linalg.norm(delta_V[0])

            #print(error)


        
        states[1] += delta_V[1]

        self.state = [states[0], states[1]] 

        tau_propagate_bis = 0.0001

        tau_span_bis = (0, tau_propagate_bis)

        t_eval_bis = [0, tau_propagate_bis]


        _, states = propagate(self.state, tau_span_bis, sat_n, lin_ctrl=None, t_eval = t_eval_bis)
        
        tau_propagate2 = 30 #continue sur l'orbite pendant 30 secondes

        delta_V_total += np.linalg.norm(delta_V[1])

        #print("Durée totale du transfert", time_transfer)

        #tau_span2 = (0, tau_propagate2)

        #t_eval2 = [0, tau_propagate2]

        #_, states = propagate(self.state, tau_span2, sat_n, lin_ctrl=0, t_eval = t_eval2)

        self.position = states[0]
        self.velocity = states[1]


        return delta_V_total, total_time, deltaV_tab
    

    def orbit_transfer_2(self, orbite_target, sat_n):

        initial_guess = np.array([0.0, 0.0, 0.0, 0])


        variables = bi_impulsive_transfer(sat_n, self.state, orbite_target, initial_guess)

        print(variables, "deltaV1")

        delta_V1 = variables[0][0]
        delta_V2 = variables[0][1]
        deltaT = variables[0]
        total_deltaV = variables[3]

        self.velocity += delta_V1

        self.state = [self.position, self.velocity]

        deltaT_2 = [0, deltaT]

        _, state1 = propagate(self.state, deltaT_2, sat_n, None)

        self.state = [state1[0], state1[1]]

        self.state[1] += delta_V2

        self.position = self.state[0]
        self.velocity = self.state[1]

        return total_deltaV
    


    ### POUR L'AFFICHAGE EN DIRECT SUR UN GRAPHIQUE ###

    def update_satellite_position(self, position):
        # Mettre à jour la position du point satellite
        self.sat_point._offsets3d = ([position[0]], [position[1]], [position[2]])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    #### #### #### ####

    def update_time_step_on_orbit(self, current_orbit, num_orbits, tau, reward, data, reward_buffer_full, time_step_on_each_orbit):
        """
        Met à jour le temps passé sur une orbite, ajuste le reward et met à jour la donnée.

        - Ajoute tau / (60 * 90) à self.time_step_on_each_orbit[current_orbit]
        - Plafonne la valeur à 1
        - Calcule un bonus exponentiellement décroissant selon le progrès gagné à ce pas
        - Réduit `data` selon l'information transmise
        - Retourne le reward ajusté et la nouvelle valeur de `data`
        """

        data_used = 0

        # Initialisation si besoin
        if not hasattr(self, 'time_step_on_each_orbit') or len(self.time_step_on_each_orbit) != num_orbits:
            self.time_step_on_each_orbit = np.zeros(num_orbits, dtype=float)

        # Incrément de temps normalisé
        increment = 0.035

        # Progrès précédent
        old_progress = self.time_step_on_each_orbit[current_orbit]

        # Mise à jour avec plafonnement à 1
        new_progress = min(1.0, old_progress + increment)

        self.time_step_on_each_orbit[current_orbit] = new_progress

        # Gain de progrès
        delta_progress = new_progress - old_progress

        #bonus = compute_bonus(new_progress, delta_progress)

        bonus = 0

        if delta_progress > 0 :
            bonus = 0.5

        if delta_progress != 0 : 
            data_used = 4.5
            self.stop = 0

        if delta_progress == 0 : 
            self.stop += 1

        # Bonus exponentiel décroissant

        # Mise à jour de la donnée (ex. : data transférée)
        
        data = min(100, data + data_used)

        return reward + bonus, data
    

    def penalize_if_idle(self, reward, threshold=0.7):
        """
        Si le reward est nul ET qu'il existe dans le tableau une valeur < threshold,
        on pénalise le reward de -1.
        """


        if reward == 0 and np.any(np.array(self.time_step_on_each_orbit) < threshold):
            self.count += 1
            reward = -1 - self.count
        return reward
    

    
    def if_orbit_fully_explored(self, threshold=0.98):
        """
        Si toutes les valeurs dans le tableau sont ≥ threshold,
        on ajoute un bonus au reward.
        """
        if np.all(np.array(self.time_step_on_each_orbit) >= threshold):
            return True
        return False
    

    def sync_flags_with_parts_visited(self, parts_visited_on_orbit, visited_flags):
        """
        Synchronise visited_flags avec parts_visited_on_orbit.

        Args:
            parts_visited_on_orbit (dict): {orbit_id: list of tuples (bool, roe_point)}
            visited_flags (dict): {orbit_id: np.array floats 0.0 or 1.0}
        """
        for orbit_id, visited_list in parts_visited_on_orbit.items():
            # extraire la première valeur (bool) de chaque tuple dans la liste
            visited_flags[orbit_id] = np.array([flag for flag, _ in parts_visited_on_orbit[orbit_id]], dtype=np.float32)



def compute_bonus(new_progress, delta_progress):
    A = 5 / delta_progress if delta_progress > 0 else 0
    B = 11.25
    return 3/2 * A * delta_progress / (1 + B * new_progress)










        


    



    


