# import os

import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    """
    Callback for logging rewards (the check is done every ``check_freq`` steps)


    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 25 episodes
                mean_reward = np.mean(y[-25:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

        return True
    

# class ActionHistoryCallback(BaseCallback):
#     """
#     Callback for getting the action history (the check is done every ``check_freq`` steps)

#     :param log_dir: (str) Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: (int)
#     """

#     def __init__(self, action_history_list, verbose=1):
#         super().__init__(verbose)
#         self.action_history_list = action_history_list
#         self.best_mean_reward = -np.inf

#     def _on_step(self) -> bool:
#         info = self.training_env.env_method("get_info")[0]
#         if info["tstep"] == 500:
#             self.action_history_list.append(info["action_history"])

#         return True