import numpy as np

class HistoryMonitor:
    def __init__(self, env) -> None:
        self.action_history = []

        self.tr_covs = np.zeros(shape=(env.steps, env.M))

    