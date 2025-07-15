import time
from stable_baselines3 import DQN
from fonctions_affichage import setup_plot, update_plot #from gym_environment import DefEnvironnement 
import matplotlib.pyplot as plt
#from gym_environment2 import DefEnvironnement2
from gym_environment import DefEnvironnement
from gym_environment_2_satellites import DefEnvironnement_n_satellites
import numpy as np

# === Load the model
model = DQN.load("dqn_meilleur_modele_5orbites_5anomlies.zip")

# === Create the environment
env = DefEnvironnement() 
obs, _ = env.reset()

# === Visualisation setup
fig, ax, point, scatter_points, battery_bar, data_bar, orbite_bars, time_text = setup_plot(env)

# === Training
done = False
total_reward = 0
step = 0

battery_levels = []
data_levels = []

while not done:
    # Predict action
    action, _ = model.predict(obs, deterministic=True)

    # Print the action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    battery_levels.append(env.battery_level)
    data_levels.append(env.data_level)

    # Total reward
    total_reward += reward

    if step % 2 == 0:
        update_plot(env, point, ax, scatter_points, battery_bar, data_bar, orbite_bars, time_text)


    # Information on each step
    print(f"[Step {step}] Reward: {reward:.2f} | Battery: {env.battery_level:.1f}% | Data: {env.data_level:.1f}%")
    print(obs[4:])
    print(action)

    time.sleep(0.05)
    step += 1


time_array = np.linspace(0, env.total_time //60, len(battery_levels))  # minutes

plt.figure(figsize=(8, 4))
plt.plot(time_array, battery_levels, color='orange')
plt.xlabel("Time (min)")
plt.ylabel("Battery Level")
plt.title("Battery Level over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot data
plt.figure(figsize=(8, 4))
plt.plot(data_levels, color='blue')
plt.xlabel("Step")
plt.ylabel("Data Level")
plt.title("Data Level over Time")
plt.grid(True)
plt.tight_layout()
plt.show()


deltaV_tab = env.deltaV_tab
total_time = env.total_time/60

n = len(deltaV_tab)
time_axis = np.linspace(0, total_time, n)  # Crée des temps uniformément espacés

plt.figure(figsize=(10, 4))
plt.vlines(time_axis, 0, deltaV_tab, color='red')
plt.xlabel("Time")
plt.ylabel("Delta-V")
plt.title("Delta-V Impulses over Time")
plt.grid(True)
plt.tight_layout()
plt.show()


# === End of the test
print("\n✅ Épisode terminé.")
print(f"🎯 Reward total: {total_reward:.2f}")
print(f"🔋 Batterie finale: {env.battery_level:.1f}%")
print(f"💾 Data finale: {env.data_level:.1f}%")
print("Nombre de transferts : ", env.nombre_transferts)
print("Cout total transferts : ", env.cout_total_transfert)
print("Total delta_V", env.cout_total_transfert / -5)
print("deltavtab", env.deltaV_tab)
print("total_time", env.total_time)


plt.ioff()
plt.show()
