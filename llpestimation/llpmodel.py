import numpy as np
from .llpproductioncrosssection import LLPProductionCrossSection

class LLPModel():
    """
    LLP model parameters and production cross section function used in LLP estimation.
    """
    def __init__(self,
                 name: str,
                 mass: float,
                 eps: float,
                 tau: float,
                 llp_xsec: LLPProductionCrossSection):
        self.name       = name                 # such as DarkLeptonicScalar, etc.
        self.mass       = mass                 # in GeV
        self.eps        = eps                  # coupling to SM
        self.tau        = tau                  # lifetime in s
        self.llp_xsec   = llp_xsec             # for computing interactions per cm
        self.unique_id  = self.get_unique_id() # string with model info besides xsec function

    def interactions_per_cm(self, energy: float) -> float:
        """
        Total cross section weighted with number density for all elements in medium.
        :param energy: Energy of muon in GeV.
        :return float: Interactions per cm.
        """
        return self.llp_xsec.interactions_per_cm(energy)

    def get_lifetime(self, gamma: float = 1.0) -> float:
        """
        Lifetime of the LLP model in lab frame for some gamma.
        :param gamma: Lorentz boost of the LLP. Given by E/m.
        :return float: Lifetime of the LLP in lab frame.
        """
        return self.tau * gamma

    def decay_factor(self, l1: float, l2: float, energy: float) -> float:
        """
        Probability to decay between lengths l1 and l2.

        Integrate exponential decay pdf between l1 and l2:
        $\int^l1_l2 1/c*gamma*tau*exp(-l/c*gamma*tau)$

        :param l1: Minimum length before decay in cm. Same as minimum detectable gap length.
        :param l2: Maximum length before decay in cm.
        :param energy: Energy of the LLP.
        :return float: Between 0-1. Fraction of the decay pdf within length l1 and l2.
        """
        c_gamma_tau = 29979245800.0 * self.get_lifetime(energy/self.mass) # c [cm/s] * gamma * tau
        prob = np.exp(-l1/c_gamma_tau) - np.exp(-l2/c_gamma_tau)
        # check for negative values
        if isinstance(prob, np.ndarray):
            prob[prob < 0] = 0.0 # if l2 is shorter than l1, return 0
        else:
            prob = max(prob, 0.0)
        return prob

    def get_unique_id(self) -> str:
        # @TODO: how to include the cross section function?
        """W
        Encodes the model in a underscore separated string.
        Used to reconstruct the LLPModel (except cross section function)."
        """
        parameters_str = [self.name, str(self.mass), str(self.eps), str(self.tau)]
        unique_id = "_".join(parameters_str)
        return unique_id

    @classmethod
    def from_unique_id(cls, unique_id: str):
        # @TODO: how to id cross section function? leave out?
        """
        Returns a new LLPModel object from a unique id.
        """
        parameters_str = unique_id.split("_")
        return cls(parameters_str[0],
                   float(parameters_str[1]),
                   float(parameters_str[2]),
                   float(parameters_str[3]),
                   None)

    def print_summary(self):
        """
        For testing purposes.
        """
        print("unique ID:", self.unique_id)
        print("Parameters of model", self.name, ":")
        print("mass", self.mass)
        print("eps", self.eps)
        print("tau", self.tau)
        print("LLPProductionCrossSection", self.llp_xsec)
        if self.llp_xsec is None:
            print("cross section is None")
        else:
            print(
                "Test LLPProductionCrossSection at 500 GeV",
                [calc_tot_xsec(500.0) for calc_tot_xsec in self.llp_xsec.func_tot_xsec_list]
            )
            print("Test interaction per cm at 500 GeV", self.interactions_per_cm(500.0))
            print("Decay factor 500 GeV 50 to 800 m", self.decay_factor(50.0, 800.0, 500.0))

