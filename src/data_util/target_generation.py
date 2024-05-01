import numpy as np
from scipy.interpolate import make_interp_spline
from typing import Optional
from numpy.typing import ArrayLike

from .cr3bp import build_taylor_cr3bp


class TargetGenerator:
    """
    Generates targets for the simulation.

    Attributes:
        catalog (np.ndarray[float]): Array containing the initial conditions of targets. Each row is an initial condition.
        periods (np.ndarray[float]): Array containing the periods of targets.
        num_options (int): Number of initial conditions in the catalog.
        dim (int): Dimension of the target state.
        mu (Tuple[float]): Tuple containing the mass ratio of the CR3BP system.
        LU (float): Unit of length in kilometers.
        TU (float): Unit of time in seconds.
        r (np.ndarray[float]): Taylor integrator object from heyokapy for integrating CR3BP equations.

    Methods:
        __init__(catalog, periods): Initializes the TargetGenerator with a catalog of targets and their periods.
        add_to_catalog(ic, period): Adds a new target to the catalog.
        gen_phased_ics(num_targets, gen_P): Provides phased initial conditions for requested targets.
        gen_phased_ics_from(x): Generates phased initial conditions from a given phase array.
        gen_state_history(catalog_ID, n_points, phase): Generates state history for a target.
        make_spline(data, periodic): Generates a spline interpolation of data.

    """
    def __init__(self, catalog: ArrayLike, periods: ArrayLike) -> None:
        """
        Initializes the TargetGenerator with a catalog of targets and their periods.

        Parameters:
            catalog: An array containing the initial conditions of targets.
            periods: An array containing the periods of targets.

        Returns:
            None

        """
        self.catalog = np.array(catalog)
        self.periods = np.array(periods)
        self.num_options = self.catalog.shape[0]
        self.dim = self.catalog.shape[1]
        self.mu = (1.215058560962404e-02,)
        self.LU = 384400
        self.TU = 3.751902619517228e+05
    
        self.r, _, _ = build_taylor_cr3bp(self.mu[0], stm=True)

    def remove_from_catalog(self, catalogID:int):
        """
        Removes a target from the catalog

        Parameters:
            catalogID (int): catalogID of the target to remove

        Returns: 
            None
        """
        self.catalog = np.delete(self.catalog, catalogID, 0)
        self.periods = np.delete(self.periods, catalogID)
        self.num_options = self.catalog.shape[0]

    def add_to_catalog(self, ic: np.ndarray[float], period: float) -> None:
        """
        Adds a new target to the catalog.

        Parameters:
            ic (np.ndarray[float]): Initial conditions of the new target.
            period (float): Period of the new target.

        Returns:
            None

        """
        self.catalog = np.vstack((self.catalog, ic))
        self.periods = np.append(self.periods, period)
        self.num_options = self.catalog.shape[0]


    def gen_phased_ics(self, catalog_ID: int, num_targets: int,  gen_P: Optional[bool] = True):
        """
        Provides phased initial conditions for requested target.

        Parameters:
            catalog_ID (int): ID of requested target
            num_targets (int): Number of initial conditions to generate.
            gen_P (Optional[bool], optional): Whether to generate covariance matrix. Defaults to True.

        Returns:
            np.ndarray: Phased initial conditions for targets.

        """
        if gen_P:
            target_P0 = np.block([[((500 / self.LU)**2) * np.eye(3), np.zeros(shape=(3,3))], [np.zeros(shape=(3,3)), ((0.001 * self.TU/self.LU)**2) * np.eye(3)]]) # 500 km uncertainty in position and 0.001 km/s uncertainty in velocity
        else:
            target_P0 = None

        targets = []

        T = self.periods[catalog_ID]
        shift = 1 / num_targets * T

        target_x = self.catalog[catalog_ID, :]

        state_hist, stm_hist = self.gen_state_history(catalog_ID, 500, phase = 0)
        spl = self.make_spline(state_hist, periodic=True)
        stm_spl = self.make_spline(stm_hist, periodic=False)
        
        targets.append({
        "state" : target_x,
        "covariance" : target_P0,
        "period" : T,
        "phase" : 0.0,
        "spline" : spl,
        "stm_spline": stm_spl})


        self.r.state[:] = np.hstack( (target_x, np.eye(self.dim).flatten()))
        self.r.time = 0
        
        for j in range(1, num_targets):

            state_hist, stm_hist = self.gen_state_history(catalog_ID, 500, phase = shift * j / T)
            target = {
                "state" : state_hist[0, 1:],
                "covariance" : target_P0,
                "period" : T,
                "phase" : shift * j / T,
                "spline": self.make_spline(state_hist, periodic=True),
                "stm_spline": self.make_spline(stm_hist, periodic=False)}
                            
            targets.append(target)

        return np.array(targets)
    
    def gen_phased_ics_from(self, x: ArrayLike):
        """
        Generates phased initial conditions from a given phase array.

        Parameters:
            x (ArrayLike): Array containing phase values for each target in the catalog.

        Returns:
            np.ndarray: Phased initial conditions for targets.

        """

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
            "spline" : spl,
            "stm_spline": stm_spl})

        return np.array(targets)
    
    def gen_state_history(self, catalog_ID: int, n_points: int, phase: Optional[float] = 0):
        """
        Generates state history for a target.

        Parameters:
            catalog_ID (int): Index of the target in the catalog.
            n_points (int): Number of points in the state history.
            phase (Optional[float], optional): Phase offset. Defaults to 0.

        Returns:
            Tuple[np.ndarray[float], np.ndarray[float]]: State and STM history.

        """

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

    
    def make_spline(self, data: np.ndarray[float], periodic: bool):
        """
        Generates a spline interpolation of data.

        Parameters:
            data (np.ndarray[float]): Array containing data points.
            periodic (bool): Whether the data represents periodic behavior.

        Returns:
            BSpline: Spline interpolation of data.

        """
        if periodic:
            x = np.append(data[:,0], data[-1, 0] + data[1,0])  # append another timestep to time grid
            y = np.append(data[:, 1:], [data[0, 1:]], axis=0)

            bspl = make_interp_spline(x, y, k=3, bc_type='periodic', axis=0)

        else:
            x = data[:,0]
            y = data[:, 1:]

            bspl = make_interp_spline(x, y, k=3, bc_type=None, axis=0)

        return bspl