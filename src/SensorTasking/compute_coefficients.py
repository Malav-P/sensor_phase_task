import numpy as np
import sys
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygmo

sys.path.append("../")

from .spacenv import SpaceEnv
from data_util.target_generation import TargetGenerator

def compute_coefficients(env):


    obs, info = env.reset()
    terminated = False

    dt = 0.1*env.observers[0].tstep

    sigma =  3 * np.pi / 180
    R_inv = 1 / sigma**2 * np.block([[np.eye(3), np.zeros(shape=(3, 3))], [np.zeros(shape=(3,3)), (0.5 * dt**2)*np.eye(3)]])


    information = np.zeros(shape=(env.maxsteps, env.observers.size, env.truths.size), dtype=float)

    for k in range(env.maxsteps):

        out = env.step()

        Hs = out[-1]


        for j,truth in enumerate(env.truths):
            stm = truth.eval_stm_spl( truth.t - truth.tstep / 2)
            stm_end = truth.eval_stm_spl(truth.period)

            # Phi(t2, t1) Phi(t1, t0)  = Phi(t2, t0)      ===>       Phi(t2, t1) = Phi(t2, t0) * Phi(t1, t0)^-1    ===== > Phi(t1, t2) = Phi(t1, t0)^-1 * Phi(t2, t0)^-1
            phi_j = stm.reshape(6, 6) @ np.linalg.inv(stm_end.reshape(6, 6))


            for i in range(env.observers.size):
                information[k, i ,j] = np.trace(phi_j.T @ Hs[i, j].T @ R_inv @ Hs[i,j] @ phi_j)

    

    return information

def solve_model(information):
    m = gp.Model("sensortask")

    # Silence model output
    m.Params.LogToConsole = 0

    # Create variables
    u = m.addMVar(shape=information.shape, vtype=GRB.BINARY, name="u")

    # Set objective
    obj = information.reshape(-1)
    m.setObjective(obj @ u.reshape(-1), GRB.MAXIMIZE)

    # observer i can only look at one target at each timestep
    m.addConstr(u.sum(axis=2) <= 1, name="row")

    m.optimize()

    return u.X, m.getObjective().getValue()



class SSA_Problem():
    def __init__(self, target_ics, target_periods,  agent_ics, agent_periods) -> None:
        tg = TargetGenerator(target_ics, periods=target_periods)
        targets = tg.gen_phased_ics([1, 1], gen_P=True)

        self.ag = TargetGenerator(agent_ics, periods = agent_periods)
        self.num_agents = len(agent_periods)
        tmp_agents = self.ag.gen_phased_ics_from([0.0] * self.num_agents)
        

        self.maxsteps = 215
        self.tstep = 0.015

        self.env = SpaceEnv(tmp_agents, targets, self.maxsteps, self.tstep)

        self.render_called = False

    def fitness(self, x):

        self._gen_env(x)
        information = compute_coefficients(self.env)
        control, objective = solve_model(information)

        return objective
    
    def get_bounds(self):
        return ([0] * self.num_agents, [1] * self.num_agents)
    
    def _gen_env(self, x):
        agents = self.ag.gen_phased_ics_from(x)
        self.env.reset_new_agents(agents=agents)

        return
    
    def render(self, x):
        self._gen_env(x)
        information = compute_coefficients(self.env)
        control, objective = solve_model(information)

        if self.render_called:
            plt.close(plt.get_fignums()[-1])
        else:
            self.render_called = True

        fig, ax = plt.subplots()

        truth_posns = np.zeros(shape=(self.env.truths.size, 2))
        obs_posns = np.zeros(shape=(self.env.observers.size, 2))
        
        # plot observer orbits
        for i, observer in enumerate(self.env.observers):
            t = np.arange(0, observer.period, observer.tstep)
            state_hist = observer.spl(t)
            ax.plot(state_hist[:,0], state_hist[:, 1], linewidth='0.5', color = 'black')
            obs_posns[i] = state_hist[0, :2]

        
        # plot target orbits 
        for i, truth in enumerate(self.env.truths):
            t = np.arange(0, truth.period, truth.tstep)
            state_hist = truth.spl(t)
            ax.plot(state_hist[:,0], state_hist[:, 1], linewidth='0.5', color = 'black')

            truth_posns[i] = state_hist[0, :2]



        truth_scat = ax.scatter(truth_posns[:,0], truth_posns[:,1], c="red")
        obs_scat = ax.scatter(obs_posns[:,0], obs_posns[:,1], c="blue")

        # connect the observer and target with a line
        ctrls = np.where(control[0] == 1)[1]
        lines = [None]*self.env.observers.size

        
        
        for i, ctrl in enumerate(ctrls):
            lines[i] = ax.plot([obs_posns[i,0], truth_posns[ctrl,0]], [obs_posns[i,1], truth_posns[ctrl,1]], linewidth = 0.2, linestyle = "--")

        def update(frame):
            # for each frame, update the data stored on each artist.
            for i, truth in enumerate(self.env.truths):
                state = truth.spl(frame*truth.tstep)
                truth_posns[i] = state[:2]

            # connect the observer and target with a line
            ctrls = np.where(control[frame] == 1)[1]

            for i, observer in enumerate(self.env.observers):
                state = observer.spl(frame*observer.tstep)
                obs_posns[i] = state[:2]
                lines[i][0].set_xdata( [obs_posns[i,0], truth_posns[ctrls[i],0]] )
                lines[i][0].set_ydata( [obs_posns[i,1], truth_posns[ctrls[i],1]] )

            # update the scatter plots
            truth_scat.set_offsets(truth_posns)
            obs_scat.set_offsets(obs_posns)

            return (truth_scat, obs_scat, *lines)


        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.env.maxsteps, interval=30)

        return ani
    
    ## TODO : (1) generate pygmo problem using this class and (2) solve the problem via GA

