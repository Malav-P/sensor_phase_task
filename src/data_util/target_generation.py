import numpy as np
from scipy.interpolate import make_interp_spline

from .cr3bp import cr3bp, jac_cr3bp, build_taylor_cr3bp


class TargetGenerator:
    def __init__(self, catalog, periods) -> None:
        self.catalog = np.array(catalog)
        self.periods = np.array(periods)
        self.num_options = self.catalog.shape[0]
        self.dim = self.catalog.shape[1]
        self.mu = (1.215058560962404e-02,)
    
        self.r, _, _ = build_taylor_cr3bp(self.mu[0], stm=True)


    def gen_phased_ics(self, num_targets,  gen_P = True):

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

            if num == 0:
                continue

            T = self.periods[i]
            shift = 1 / num * T

            target_x = self.catalog[i, :]

            state_hist, stm_hist = self.gen_state_history(i, 500, phase = 0)
            spl = self.make_spline(state_hist, periodic=True)
            stm_spl = self.make_spline(stm_hist, periodic=False)
            
            targets.append({
            "state" : target_x,
            "covariance" : target_P_0,
            "period" : T,
            "phase" : 0.0,
            "f": cr3bp,
            "jac" : jac_cr3bp,
            "f_params": self.mu,
            "jac_params" : self.mu,
            "spline" : spl,
            "stm_spline": stm_spl})


            self.r.state[:] = np.hstack( (target_x, np.eye(self.dim).flatten()))
            self.r.time = 0
            
            for j in range(1, num):

                state_hist, stm_hist = self.gen_state_history(i, 500, phase = shift * j / T)
                target = {
                    "state" : state_hist[0, 1:],
                    "covariance" : target_P_0,
                    "period" : T,
                    "phase" : shift * j / T,
                    "f": cr3bp,
                    "jac" : jac_cr3bp,
                    "f_params": self.mu,
                    "jac_params" : self.mu,
                    "spline": self.make_spline(state_hist, periodic=True),
                    "stm_spline": self.make_spline(stm_hist, periodic=False)}
                                
                targets.append(target)

        return np.array(targets)
    
    def gen_phased_ics_from(self, x):

            targets = []

            for i, phase in enumerate(x):

                T = self.periods[i]

                target_x = self.catalog[i, :]

                state_hist, stm_hist = self.gen_state_history(i, 500, phase=phase)
                spl = self.make_spline(state_hist, periodic=True)
                stm_spl = self.make_spline(stm_hist, periodic=False)
                
                targets.append({
                "state" : target_x,
                "covariance" : None,
                "period" : T,
                "phase" : phase,
                "f": cr3bp,
                "jac" : jac_cr3bp,
                "f_params": self.mu,
                "jac_params" : self.mu,
                "spline" : spl,
                "stm_spline": stm_spl})

            return np.array(targets)
    
    def gen_state_history(self, catalog_ID, n_points, phase = 0):

        tt = np.linspace(0, self.periods[catalog_ID], n_points)
        state_history = np.zeros(shape=(n_points, 1 + self.dim))
        stm_history = np.zeros(shape=(n_points, 1 + self.dim**2))

        ic = self.catalog[catalog_ID]

        state_history[:, 0] = tt
        stm_history[:,0] = tt
    
        self.r.state[:] = np.hstack((ic, np.eye(self.dim).flatten()))
        self.r.propagate_for(delta_t = phase * self.periods[catalog_ID])


        self.r.time = 0
        state_history[0, 1:] = self.r.state[:self.dim]
        stm_history[0, 1:] = self.r.state[self.dim:]

        out = self.r.propagate_grid(tt)

        state_history[:, 1:] = out[-1][:, :self.dim]
        stm_history[:, 1:] = out[-1][:, self.dim:]

        return state_history, stm_history
    
    # def gen_stm_history(self, catalog_ID, n_points):

    #     tt = np.linspace(0, self.periods[catalog_ID], n_points)
    #     data = np.zeros(shape=(n_points, 1 + self.dim*self.dim))

    #     ic = np.hstack((self.catalog[catalog_ID], np.eye(self.dim).flatten()))

    #     data[:, 0] = tt
    #     data[0, 1:] = ic[self.dim:]

    #     self.r.time = 0
    #     self.r.state[:] = ic

    #     out = self.r.propagate_grid(tt)

    #     data[:, 1:] = out[-1][:, self.dim:]

    #     return data
    
    # def normalize_data(self, data, LU = 1.0, TU = 1.0, center = np.array([0, 0, 0])):
    #     norm_data = np.copy(data)
    #     norm_data[:, 0] = (data[:,0] - data[0, 0]) / TU
    #     norm_data[:, [1, 2, 3]] = data[:, [1, 2, 3]] / LU + np.tile(center, (data.shape[0], 1))
    

    #     return norm_data
    
    def make_spline(self, data, periodic):
        if periodic:
            x = np.append(data[:,0], data[-1, 0] + data[1,0])
            y = np.append(data[:, 1:], [data[0, 1:]], axis=0)

            bspl = make_interp_spline(x, y, k=3, bc_type='periodic', axis=0)

        else:
            x = data[:,0]
            y = data[:, 1:]

            bspl = make_interp_spline(x, y, k=3, bc_type=None, axis=0)

        return bspl