import numpy as np

class STDP:
    def __init__(self, tau_plus=50.0, tau_minus=50.0, A_plus=0.01, A_minus=0.01):
        """
        Initialize Spike-Timing-Dependent Plasticity (STDP) parameters.
        
        Parameters:
        - tau_plus (float): Time constant for Long-Term Potentiation (LTP) in ms.
        - tau_minus (float): Time constant for Long-Term Depression (LTD) in ms.
        - A_plus (float): Maximum potentiation factor for LTP.
        - A_minus (float): Maximum depression factor for LTD.
        """
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus

    def update_weight(self, delta_t):
        """
        Update the synaptic weight based on the time difference between pre- and postsynaptic spikes.
        
        Parameters:
        - delta_t (float): Time difference (pre - post) in ms.
        
        Returns:
        - float: The change in synaptic weight.
        """
        if delta_t > 0:
            # LTP: Presynaptic spike occurs before postsynaptic spike
            delta_w = self.A_plus * np.exp(-delta_t / self.tau_plus)
        else:
            # LTD: Presynaptic spike occurs after postsynaptic spike
            delta_w = -self.A_minus * np.exp(delta_t / self.tau_minus)
        
        return delta_w
