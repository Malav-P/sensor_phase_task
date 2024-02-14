from abc import ABC, abstractmethod
from scipy.integrate import ode

class State(ABC):
    def __init__(self, x0, tstep):
        self.ic = x0
        self.x = x0
        self.t = 0
        self.shape = x0.shape
        self.tstep = tstep

    @abstractmethod
    def propagate(self, steps = 1):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_initial_value(self, y, t = 0):
        pass

class Dynamics(State):
    def __init__(self, x0, tstep, f, jac=None, f_params=None, jac_params=None):
        super().__init__(x0=x0, tstep=tstep)
        self.r = ode(f, jac).set_integrator('dop853')
        
        if f_params is not None:
            self.r.set_f_params(*f_params)
        if jac_params is not None:
            self.r.set_jac_params(*jac_params)

        self.r.set_initial_value(x0, 0)

    def propagate(self, steps=1):
        self.x = self.r.integrate(self.r.t + self.tstep*steps)
        self.t += self.tstep * steps
        return self.x
    
    def reset(self):
        self.x = self.ic
        self.t = 0
        self.r.set_initial_value(self.ic, 0)

    def set_initial_value(self, y, t=0):
        self.x = y
        self.t = t
        self.r.set_initial_value(y, t=t)
        


class Spline(State):
    def __init__(self, x0, tstep, spl):
        super().__init__(x0, tstep)
        self.spl = spl

    def propagate(self, steps=1):
        self.t += steps*self.tstep
        self.x = self.spl(self.t)
        return self.x
    
    def reset(self):
        self.x = self.ic
        self.t = 0

    def set_initial_value(self, y, t=0):
        self.x = y
        self.t = t


class Analytic(State):
    def __init__(self, x0, tstep, functions):
        super().__init__(x0, tstep)
        self.functions = functions

        assert x0.size == functions.size, "number of variables and functions mismatch"

    def propagate(self, steps=1):
        self.t += steps*self.tstep
        for i in range(self.x.size):
            self.x[i] = self.functions[i](self.t)

        return self.x
    
    def reset(self):
        self.x = self.ic
        self.t = 0
    
    def set_initial_value(self, y, t=0):
        self.x = y
        self.t = t
