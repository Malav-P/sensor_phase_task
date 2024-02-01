# Continuous-Discrete Kalman Filter using MultiSensor Information Filter (Inverse Covariance Filter)
# Adapted from https://web.archive.org/web/20170217222640/http://www.acfr.usyd.edu.au/pdfs/training/IntroDataFusionSlides.pdf
#         and  https://en.wikipedia.org/wiki/Extended_Kalman_filter

import numpy as np
from .state import Dynamics

class KalmanFilter:
    def __init__(self, timestep, xof, Pof, func, jac, params):
        self.ic = xof
        self.ic_cov = Pof
        self.xf = xof
        self.Pf = Pof
        self.xa = xof
        self.Pa = Pof
        self.dim = xof.size
        # self.Q = np.zeros(shape=(self.dim, self.dim))
        self.Q = 0.001 * np.eye(self.dim)

        def _kalman_system(t, u):
            d = self.dim

            x = u[:d]
            P = np.reshape(u[d:], newshape=(d, d))

            F = jac(t, x, *params)

            dx = func(t, x, *params)
            dP = F @ P + P @ F.T + self.Q

            du = np.hstack((dx, np.ravel(dP)))

            return du
        
        u0 = np.hstack((self.xa, np.ravel(self.Pa)))
        self.fcint = Dynamics(u0, timestep, _kalman_system)

    def propagate(self, Z, R_invs):
        self._forecast(steps=1)
        self._update(Z = Z, R_invs = R_invs)

    def _update(self, Z, R_invs, H_s = None):
        if Z is None:
            self.Pa = self.Pf
            self.xa = self.xf

        else:
            n_obs = R_invs.shape[2]
            Pf_inv = np.linalg.inv(self.Pf)

            if H_s is None:
                information_matrices = R_invs
                self.Pa = np.linalg.inv(Pf_inv + np.sum( information_matrices, axis=2 ))
                self.xa = self.Pa @ (Pf_inv @ self.xf + sum( [R_invs[:, :, i] @ Z[:,i] for  i in range(n_obs)] ))

            else:  
                information_matrices = np.array([H_s[:,:,i].T @ R_invs[:,:,i] @ H_s[:,:,i] for i in range(n_obs)])
                self.Pa = np.linalg.inv(Pf_inv + np.sum( information_matrices, axis=2 ))
                self.xa = self.Pa @ (Pf_inv @ self.xf + sum( [H_s[:,:,i].T @ R_invs[:, :, i] @ Z[:,i] for  i in range(n_obs)] ))
            
        
        self.Pa = (self.Pa + self.Pa.T)/2 # ensure posterior covariance symmetric

        assert np.allclose(self.Pa, self.Pa.T, rtol=1e-5, atol=1e-8)

    
    def _forecast(self, steps):
        ua = np.hstack((self.xa, np.ravel(self.Pa)))
        self.fcint.set_initial_value(ua, t=self.fcint.t)

        uf = self.fcint.propagate(steps=steps)
        self.xf = uf[:self.dim]
        self.Pf = np.reshape(uf[self.dim:], newshape=(self.dim, self.dim))

        self.Pf = 0.5 * (self.Pf + self.Pf.T) # ensure symmetric covariance.

        assert np.allclose(self.Pf, self.Pf.T, rtol=1e-5, atol=1e-8)

        return
    
    def reset(self):
        self.xf = self.ic
        self.Pf = self.ic_cov
        self.xa = self.ic
        self.Pa = self.ic_cov
        self.fcint.reset()
