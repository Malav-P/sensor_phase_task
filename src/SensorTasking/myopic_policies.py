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

def func(x, y, env):
    n_observers = x.size
    assert y.shape[1] == n_observers, "control not given for all observers!"

    obs, info = env.reset()
    for observer, phase in zip(env.observers, x):
        observer.set_initial_value(phase)

    terminated = False
    
    cum_reward = 0

    for control in y:
        
        obs, reward, terminated, tmp, info = env.step(np.array([control]))

        cum_reward += reward

        if terminated:
            print("terminated episode")
            break

    return cum_reward



            