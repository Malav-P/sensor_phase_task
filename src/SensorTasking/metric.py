from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    def __init__(self, params, states=None):
        self.params = params
        self.states = states

    @abstractmethod
    def compute(observer, truth):
        pass

class ApparentMag(Metric):
    def __init__(self, params, sun):
        super().__init__(params, list(sun))
    
    def compute(self, observer, truth):
        ms = self.params["ms"]
        aspec = self.params["aspec"]
        adiff = self.params["adiff"]
        d = self.params["d"]

        rS = self.states[0].x[:3]
        rO = observer.x[:3]
        rT = truth.x[:3]


        rOT = rT - rO
        rST = rT - rS

        zeta = np.linalg.norm(rOT)
        psi = np.arctan2(np.linalg.norm(np.cross(rOT, rST)), np.dot(rOT, rST))
        pdiff = (2/(3*np.pi)) * (np.sin(psi) + (np.pi - psi)*np.cos(psi))

        if self._deadzone(observer, truth, body="Earth") or self._deadzone(observer, truth, body="Moon"):
            apmag = np.finfo(np.float64).max

        else:
            apmag = ms - 2.5 * np.log10((d**2)/(zeta**2)*(aspec/4 + adiff*pdiff))

        return apmag
    
    def _deadzone(self, observer, truth, body):
        alpha = self.params["rearth"]

        rT = truth.x[:3]
        rO = observer.x[:3]
        if body == "Earth":
            rE = [-self.params["mu"], 0, 0]
        elif body == "Moon":
            rE = [1- self.params["mu"], 0, 0]
        else:
            ValueError("argument body must be either Earth or Moon")

        rOE = rE - rO

        w1 = np.array([0, -rOE[2], rOE[1]]) / np.linalg.norm(np.array([0, -rOE[2], rOE[1]]))
        b1 = np.dot(w1, rE)

        w2 = np.cross(rOE, w1) / np.linalg.norm(rOE)
        b2 = np.dot(w2, rE)

        w3 = rOE / np.linalg.norm(rOE)
        b3 = np.dot(w3, rE)

        if np.abs(np.dot(w1, rT) - b1) <= alpha and np.abs(np.dot(w2, rT)-b2) <= alpha and np.dot(w3, rT) - b3 > 0:
            return True
        else:
            return False
