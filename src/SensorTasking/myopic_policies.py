import numpy as np
def run_myopic_policy(env):

    obs, info = env.reset()
    terminated = False

    cum_reward = 0

    while not terminated:
        most_uncertain_target, uncertainty = (1, 0)

        for i, kalman_object in enumerate(env.kalman_objects):
            tar_uncertainty = np.trace(kalman_object.P)

            if tar_uncertainty > uncertainty:
                most_uncertain_target, uncertainty = (i+1, tar_uncertainty)

        obs, reward, terminated, tmp, info = env.step(most_uncertain_target)

        cum_reward += reward

    return cum_reward

            