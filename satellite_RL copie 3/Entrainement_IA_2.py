import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from gym_environment import DefEnvironnement  # Ton environnement Gym encapsulé
from stable_baselines3.dqn import MlpPolicy
import torch
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# Créer l'environnement
env = DefEnvironnement()

# Vérifier sa validité (optionnel)
check_env(env, warn=True)

class TransferLoggerEpisodeCallback(BaseCallback):
    def __init__(self, log_every_n_episodes=1):
        super().__init__()
        self.episode_counter = 0
        self.log_every_n = log_every_n_episodes

    def _on_step(self) -> bool:
        # Récupérer l'environnement
        env = self.training_env.envs[0].env.unwrapped
        dones = self.locals.get("dones", [False])

        if dones[0]:

            self.episode_counter += 1

            if self.episode_counter % self.log_every_n == 0:
                # L’épisode est terminé, on log les valeurs
    
                self.logger.record("episode/cout_total_transfert", env.last_episode_cout)
                self.logger.record("episode/nombre_de_transferts", env.last_episode_transferts)
                self.logger.record("episode/Minimum_battery", env.real_result_battery)
                self.logger.record("episode/Remplissage maximum du buffer", env.real_result_buffer)

        return True

policy_kwargs = {
    "net_arch": [256, 256]
    } # [256, 256, 128]

# Créer l'agent DQN
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=3000,
    batch_size=256,
    train_freq=3,
    target_update_interval=30000,
    exploration_fraction=0.6,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./dqn_satellite_logs/", 
)

model.learn(
    total_timesteps=4_000_000,
    callback=TransferLoggerEpisodeCallback(),
    log_interval=1,
)



# Sauvegarder le modèle
model.save("dqn_satellite_model_4_5orbites")
