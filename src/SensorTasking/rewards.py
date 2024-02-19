import numpy as np

# reward proportional to the sum of trace covariance of all VISIBLE TARGETS
def reward1(self):
        reward = 0

        if (self.prev_action not in self.prev_available_actions) and (self.prev_action is not None):
            reward -= 10

        for kalman_object in self.kalman_objects[[x-1 for x in self.prev_available_actions]]:
            reward -= 10*np.trace(kalman_object.P)

        return reward

# reward proportional to trace reduction
def reward2(self):
     
        reward = 0

        if (self.prev_action not in self.prev_available_actions) and (self.prev_action is not None):
            reward -= 10

        for kalman_object in self.kalman_objects[[x-1 for x in self.prev_available_actions]]:
            reward += 100 * (np.trace(kalman_object.P_prev) - np.trace(kalman_object.P))

        return reward

        
     