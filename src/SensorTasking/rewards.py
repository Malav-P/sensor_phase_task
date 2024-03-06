import numpy as np

# reward proportional to the sum of trace covariance of all VISIBLE TARGETS
def reward1(self):
    reward = 0

    if np.any([self.prev_action[i] not in self.available_actions[i] for i in range(self.N)]):
        reward -= 10

    for kalman_object in self.kalman_objects:
        if np.any([self.obs_model.is_visible(kalman_object,observer) for observer in self.observers]):
            reward -= 10*np.trace(kalman_object.P)

    return reward

# reward proportional to trace reduction
def reward2(self):
     
    reward = 0

    if np.any([self.prev_action[i] not in self.available_actions[i] for i in range(self.N)]):
        reward -= 10

    for kalman_object in self.kalman_objects:
        if np.any([self.obs_model.is_visible(kalman_object,observer) for observer in self.observers]):
            reward += 100 * (np.trace(kalman_object.P_prev) - np.trace(kalman_object.P))

    return reward

        
     