from abc import ABC, abstractmethod
from scipy.integrate import ode
from scipy.interpolate import BSpline
import numpy as np
from typing import Optional, Callable, List


class State(ABC):
    """
    Abstract base class representing a state in a dynamical system.

    Attributes:
        ic: Initial condition of the state.
        x: Current state.
        t: Current time.
        shape: Shape of the state.
        tstep: Time step size.

    Methods:
        __init__(x0, tstep): Initializes the State with initial condition and time step.
        propagate(steps): Abstract method to propagate the state.
        reset(): Abstract method to reset the state.

    """
    def __init__(self, x0: np.ndarray[float], tstep: float):
        """
        Initializes the State with initial condition and time step.

        Parameters:
            x0: Initial condition.
            tstep: Time step size.

        """
        self.ic = x0
        self.x = x0
        self.t = 0
        self.shape = x0.shape
        self.tstep = tstep

    @abstractmethod
    def propagate(self, steps: Optional[int] = 1):
        """
        Propagates the state forward by the specified number of steps.

        Parameters:
            steps (int): Number of steps to propagate.

        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the state to its initial condition.

        """
        pass

class Dynamics(State):
    """
    Represents a dynamical state in the system.

    Parameters:
        x0 (np.ndarray[float]): Initial condition.
        tstep (float): Time step size.
        f (Callable): Function defining the dynamics.
        jac (Optional[Callable]): Jacobian of the dynamics (optional).
        f_params (Optional[tuple[float]]): Additional parameters for the dynamics function (optional).
        jac_params (Optional[tuple[float]]): Additional parameters for the Jacobian function (optional).

    Attributes:
        r: ODE integrator.

    Methods:
        __init__(x0, tstep, f, jac=None, f_params=None, jac_params=None): Initializes the Dynamics state.
        propagate(steps): Propagates the state forward by the specified number of steps.
        reset(): Resets the state to its initial condition.
        set_initial_value(y, t=0): Sets the initial value of the state.

    """
    def __init__(self,
                x0: np.ndarray[float],
                tstep: float,
                f: Callable[[np.ndarray], np.ndarray],
                jac: Optional[Callable] = None,
                f_params: Optional[tuple[float]] = None,
                jac_params: Optional[tuple[float]] = None):

        super().__init__(x0=x0, tstep=tstep)
        self.r = ode(f, jac).set_integrator('dop853')
        
        if f_params is not None:
            self.r.set_f_params(*f_params)
        if jac_params is not None:
            self.r.set_jac_params(*jac_params)

        self.r.set_initial_value(x0, 0)

    def propagate(self, steps: Optional[int] = 1):
        """
        Propagates the state forward by the specified number of steps.

        Parameters:
            steps (int): Number of steps to propagate.

        Returns:
            np.ndarray: The propagated state.

        """
        self.x = self.r.integrate(self.r.t + self.tstep*steps)
        self.t += self.tstep * steps
        return self.x
    
    def reset(self):
        """
        Resets the state to its initial condition.

        """
        self.x = self.ic
        self.t = 0
        self.r.set_initial_value(self.ic, 0)

    def set_initial_value(self, y: np.ndarray[float], t: Optional[float] = 0):
        """
        Sets the initial value of the state.

        Parameters:
            y (np.ndarray): Initial value.
            t (float): Initial time (default is 0).

        """
        self.x = y
        self.t = t
        self.r.set_initial_value(y, t=t)
        


class Spline(State):
    """
    Represents a state defined by a spline.

    Parameters:
        tstep (float): Time step size.
        spl (BSpline): Spline function.
        stm_spl (BSpline): Spline function for state transition matrix.
        period (float): Period of the spline.

    Attributes:
        period (float): Period of the spline.
        spl (BSpline): Spline function.
        stm_spl (BSpline): Spline function for state transition matrix.

    Methods:
        __init__(tstep, spl, stm_spl, period): Initializes the Spline state.
        propagate(steps): Propagates the state forward by the specified number of steps.
        reset(): Resets the state to its initial condition.
        eval_stm_spl(t): Evaluates the spline function for the state transition matrix.

    """
    def __init__(self, tstep: float, spl: BSpline, stm_spl: BSpline, period: float):

        self.period = period
        self.spl = spl
        self.stm_spl = stm_spl

        x0 = spl(0)
    
        super().__init__(x0, tstep)
        
    def propagate(self, steps: Optional[int] = 1):
        """
        Propagates the state forward by the specified number of steps.

        Parameters:
            steps (int): Number of steps to propagate.

        Returns:
            np.ndarray: The propagated state.

        """
        self.t += steps*self.tstep
        self.x = self.spl(self.t)
        return self.x
    
    def reset(self):
        """
        Resets the state to its initial condition.

        """
        self.x = self.ic
        self.t = 0

    def eval_stm_spl(self, t: float):
        """
        Evaluates the state transition matrix (STM) at the requested time.

        Parameters:
            t (float): time at which to evaluate the STM.

        Returns:
            np.ndarray: The evaluated STM as a flattened array.
        
        Raises:
            ValueError: If the requested time exceeds the period.

        Notes:
            - The returned STM will be a flattened ndarray. Use np.reshape to arrange elements into a proper matrix.

        """
        if t > self.period:
            raise ValueError("requested eval time exceeds the propagated time for STM")
        else:
            return self.stm_spl(t)


class Analytic(State):
    """
    Represents a state with analytic functions to propagate the state.

    Parameters:
        x0 (np.ndarray[float]): The initial state vector.
        tstep (float): The time step for propagation.
        functions (List[Callable[[float], float]]): List of analytic functions representing state evolution.

    Raises:
        ValueError: If the number of variables and functions mismatch.

    Attributes:
        functions (List[Callable[[float]]): List of analytic functions representing state evolution.

    Methods:
        propagate(steps=1): Propagates the state forward by the specified number of time steps.
        reset(): Resets the state to its initial configuration.

    """
    def __init__(self,
                x0: np.ndarray[float],
                tstep: float,
                functions: List[Callable[[float], float]]):
        
        super().__init__(x0, tstep)
        self.functions = functions

        if x0.size != functions.size:
            raise ValueError("Number of variables and functions mismatch.")

    def propagate(self, steps: Optional[int] = 1):
        """
        Propagates the state forward by the specified number of time steps.

        Parameters:
            steps (int, optional): The number of time steps to propagate. Defaults to 1.

        Returns:
            np.ndarray[float]: The propagated state vector.
        """
        self.t += steps*self.tstep
        for i in range(self.x.size):
            self.x[i] = self.functions[i](self.t)

        return self.x
    
    def reset(self):
        """
        Resets the state to its initial configuration.
        """
        self.x = self.ic
        self.t = 0