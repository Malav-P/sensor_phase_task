import numpy as np
from scipy.integrate import ode
from scipy.interpolate import make_interp_spline

from .cr3bp import cr3bp, jac_cr3bp

class TargetGenerator:
    def __init__(self, catalog, periods) -> None:
        self.catalog = np.array(catalog)
        self.periods = np.array(periods)
        self.num_options = self.catalog.shape[0]
        self.dim = self.catalog.shape[1]

        mu = (1.215058560962404e-02,)

        self.r = ode(cr3bp, jac_cr3bp).set_integrator('dop853').set_f_params(*mu).set_jac_params(*mu)


    def gen_phased_ics(self, num_targets, gen_P = True):

        LU = 384400
        TU = 3.751902619517228e+05

        if isinstance(num_targets, list):
            num_targets = np.array(num_targets)
        elif isinstance(num_targets, int):
            num_targets = num_targets * np.ones(self.num_options, dtype=int)
        
        if gen_P:
            target_P_0 = np.block([[((500 / LU)**2) * np.eye(3), np.zeros(shape=(3,3))], [np.zeros(shape=(3,3)), ((0.001 * TU/LU)**2) * np.eye(3)]]) # 500 km uncertainty in position and 0.001 km/s uncertainty in velocity
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
    
    def gen_state_history(self, catalog_ID, n_points):

        tt = np.linspace(0, self.periods[catalog_ID], n_points)
        data = np.zeros(shape=(n_points, 1 + self.dim))

        
        ic = self.catalog[catalog_ID]

        data[:, 0] = tt
        data[0, 1:] = ic

        self.r.set_initial_value(ic, t = 0)

        for i in range(1, tt.size):
            data[i, 1:] = self.r.integrate(tt[i])


        return data
    
    def normalize_data(self, data, LU = 1.0, TU = 1.0, center = np.array([0, 0, 0])):
        norm_data = np.copy(data)
        norm_data[:, 0] = (data[:,0] - data[0, 0]) / TU
        norm_data[:, [1, 2, 3]] = data[:, [1, 2, 3]] / LU + np.tile(center, (data.shape[0], 1))
    

        return norm_data
    
    def make_spline(self, data):
        x = np.append(data[:,0], data[-1, 0] + data[1,0])
        y = np.append(data[:, [1 ,2 ,3]], [data[0, [1, 2, 3]]], axis=0)

        bspl = make_interp_spline(x, y, k=3, bc_type='periodic', axis=0)

        return bspl



            

            





