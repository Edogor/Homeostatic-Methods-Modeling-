import numpy as np

class LIFNeuron:
    def __init__(self, tau_m=25.0, v_rest=-65.0, v_reset=-65.0, v_threshold=-50.0, g_L=0.007, e_L=-65.0):
        """
        Initialize a Leaky Integrate-and-Fire (LIF) neuron.
        
        Parameters:
        - tau_m (float): Membrane time constant in ms.
        - v_rest (float): Resting membrane potential in mV.
        - v_reset (float): Reset potential after a spike in mV.
        - v_threshold (float): Firing threshold in mV.
        - g_L (float): Leak conductance in µS.
        - e_L (float): Leak reversal potential (same as resting potential) in mV.
        """
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.g_L = g_L
        self.e_L = e_L
        self.v_m = self.v_rest  # Initialize membrane potential at resting potential
        self.spike = False
        self.spike_count = 0  # Track spike count for Homeostatic Intrinsic Plasticity (HIP)

    def reset(self):
        """Reset the neuron to its resting potential after a spike."""
        self.v_m = self.v_reset
        self.spike = False

    def update(self, i_syn, dt):
        """
        Update the membrane potential based on the synaptic input current.
        
        Parameters:
        - i_syn (float): Synaptic current in nA.
        - dt (float): Time step in ms.
        
        Returns:
        - bool: True if the neuron fires a spike, False otherwise.
        """
        # Update membrane potential using the LIF equation
        dv = (-(self.v_m - self.e_L) + i_syn / self.g_L) / self.tau_m * dt
        self.v_m += dv
        
        # Check if the neuron fires a spike
        if self.v_m >= self.v_threshold:
            self.spike = True
            self.spike_count += 1
            self.reset()  # Reset potential after spike
            return True

        self.spike = False
        return False