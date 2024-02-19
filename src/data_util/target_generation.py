import numpy as np
from scipy.integrate import ode

from .cr3bp import cr3bp, jac_cr3bp

class TargetGenerator:
    def __init__(self, catalog, periods) -> None:
        self.catalog = np.array(catalog)
        self.periods = np.array(periods)
        self.num_options = self.catalog.shape[0]
        self.dim = self.catalog.shape[1]

        mu = (1.215058560962404e-02,)

        self.r = ode(cr3bp, jac_cr3bp).set_integrator('dop853').set_f_params(*mu).set_jac_params(*mu)


    def gen_phased_ics(self, num_targets, stochastic = True):

        if isinstance(num_targets, list):
            num_targets = np.array(num_targets)
        elif isinstance(num_targets, int):
            num_targets = num_targets * np.ones(self.num_options, dtype=int)
        
        if stochastic:
            target_P_0 = 0.001 * np.eye(6)
        else:
            target_P_0 = None

        assert num_targets.size <= self.num_options, "length of num_targets must be <= number of catalog objects"

        targets = []

        for i, num in enumerate(num_targets):

            T = self.periods[i]
            shift = 1 / num * T

            target_x = self.catalog[i, :]
            
            targets.append({
            "state" : target_x,
            "covariance" : target_P_0,
            "f": cr3bp,
            "jac" : jac_cr3bp,
            "f_params": self.r.f_params,
            "jac_params" : self.r.jac_params})


            self.r.set_initial_value(target_x, t = 0)
            
            for i in range(1, num):
            
                target_x = self.r.integrate(t=self.r.t + shift)

                target = {
                    "state" : target_x,
                    "covariance" : target_P_0,
                    "f": cr3bp,
                    "jac" : jac_cr3bp,
                    "f_params": self.r.f_params,
                    "jac_params" : self.r.jac_params}
                
                targets.append(target)

        return np.array(targets)
            

            





