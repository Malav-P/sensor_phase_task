# Continuous-Discrete Kalman Filter using MultiSensor Information Filter (Inverse Covariance Filter)
# Adapted from https://web.archive.org/web/20170217222640/http://www.acfr.usyd.edu.au/pdfs/training/IntroDataFusionSlides.pdf
#         and  https://en.wikipedia.org/wiki/Extended_Kalman_filter

import numpy as np
from .state import Dynamics

class KalmanFilter:
    def __init__(self, timestep, xof, Pof, func, jac, f_params=None, jac_params=None):
        self.ic = xof
        self.ic_cov = Pof
        self.x = xof
        self.P = Pof
        self.P_prev = Pof
        self.dim = xof.size
        self.Q = 0.0001 * np.eye(self.dim)


        def _kalman_system(t, u):
            d = self.dim

            x = u[:d]
            P = np.reshape(u[d:], newshape=(d, d))

            F = jac(t, x, *jac_params)

            dx = func(t, x, *f_params)
            dP = F @ P + P @ F.T + self.Q

            du = np.hstack((dx, np.ravel(dP)))

            return du
        
        u0 = np.hstack((self.x, np.ravel(self.P)))
        self.fcint = Dynamics(u0, timestep, _kalman_system)

    def propagate(self, Z, R_invs):
        self.P_prev = self.P

        self._forecast(steps=1)
        self._update(Z = Z, R_invs = R_invs)

    def _update(self, Z, R_invs, H_s = None):
        if Z is None:
            pass
        else:
            n_obs = R_invs.shape[2]
            P_prior_inv = np.linalg.inv(self.P)

            if H_s is None:
                information_matrices = R_invs
                self.P = np.linalg.inv(P_prior_inv + np.sum( information_matrices, axis=2 ))
                self.x = self.P @ (P_prior_inv @ self.x + sum( [R_invs[:, :, i] @ Z[:,i] for  i in range(n_obs)] ))

            else:  
                information_matrices = np.array([H_s[:,:,i].T @ R_invs[:,:,i] @ H_s[:,:,i] for i in range(n_obs)])
                self.P = np.linalg.inv(P_prior_inv + np.sum( information_matrices, axis=2 ))
                self.x = self.P @ (P_prior_inv @ self.x + sum( [H_s[:,:,i].T @ R_invs[:, :, i] @ Z[:,i] for  i in range(n_obs)] ))
            
        
        self.P = (self.P + self.P.T)/2 # ensure posterior covariance symmetric

        assert np.allclose(self.P, self.P.T, rtol=1e-5, atol=1e-8)

    
    def _forecast(self, steps):
        ua = np.hstack((self.x, np.ravel(self.P)))
        self.fcint.set_initial_value(ua, t=self.fcint.t)

        uf = self.fcint.propagate(steps=steps)
        self.x = uf[:self.dim]
        self.P = np.reshape(uf[self.dim:], newshape=(self.dim, self.dim))

        self.P = 0.5 * (self.P + self.P.T) # ensure symmetric covariance.

        assert np.allclose(self.P, self.P.T, rtol=1e-5, atol=1e-8)

        return
    
    def reset(self):
        self.x = self.ic
        self.P = self.ic_cov
        self.P_prev = self.ic_cov
        self.fcint.reset()