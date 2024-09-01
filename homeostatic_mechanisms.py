import numpy as np
import matplotlib.pyplot as plt

class HomeostaticMechanisms:
    def __init__(self, use_weight_normalization=False, use_synaptic_scaling=False, target_synaptic_sum=1.0, use_hip=False, target_spike_rate=0.008, learning_rate=4, use_lateral_inhibition=False, lateral_inhibition_strength=3.6, use_global_inhibition=False, global_inhibition_threshold=3, global_inhibition_scale=0.8):
        """
        Initialize various homeostatic mechanisms for the neural network.
        
        Parameters:
        - use_weight_normalization (bool): Enable weight normalization.
        - use_synaptic_scaling (bool): Enable synaptic scaling.
        - target_synaptic_sum (float): Target sum of incoming synaptic weights for scaling.
        - use_hip (bool): Enable Homeostatic Intrinsic Plasticity (HIP).
        - target_spike_rate (float): Target spike rate for HIP.
        - learning_rate (float): Learning rate for HIP threshold adjustment.
        - use_lateral_inhibition (bool): Enable lateral inhibition.
        - lateral_inhibition_strength (float): Strength of lateral inhibition.
        - use_global_inhibition (bool): Enable global inhibition.
        - global_inhibition_threshold (float): Spike threshold for triggering global inhibition.
        - global_inhibition_scale (float): Scaling factor for global inhibition.
        """
        self.use_weight_normalization = use_weight_normalization
        self.use_synaptic_scaling = use_synaptic_scaling
        self.use_hip = use_hip    
        self.use_lateral_inhibition = use_lateral_inhibition
        self.use_global_inhibition = use_global_inhibition
        self.target_spike_rate = target_spike_rate
        self.learning_rate = learning_rate
        self.target_synaptic_sum = target_synaptic_sum
        self.lateral_inhibition_strength = lateral_inhibition_strength
        self.global_inhibition_threshold = global_inhibition_threshold
        self.global_inhibition_scale = global_inhibition_scale
        self.v_threshold_history = []  # List to store v_threshold values over time
        self.inhibited_neurons = {}  # Dictionary to track inhibited neurons and inhibition times

    def apply_weight_normalization(self, weights):
        """
        Normalize synaptic weights to prevent runaway excitation or inhibition.
        
        Parameters:
        - weights (ndarray): Synaptic weights matrix.
        
        Returns:
        - ndarray: Normalized weights.
        """
        if self.use_weight_normalization:
            max_weight = np.max(weights)
            if max_weight > 0:  # Avoid division by zero
                weights = weights / max_weight
        return weights

    def apply_synaptic_scaling(self, weights):
        """
        Apply synaptic scaling to maintain synaptic strength within a target range.
        
        Parameters:
        - weights (ndarray): Synaptic weights matrix.
        
        Returns:
        - ndarray: Scaled weights.
        """
        if self.use_synaptic_scaling:
            for i in range(weights.shape[1]):  # Iterate over neurons (incoming weights)
                sum_weights = np.sum(weights[:, i])
                if sum_weights > 0:
                    scaling_factor = self.target_synaptic_sum / sum_weights
                    weights[:, i] *= scaling_factor  # Scale all incoming weights of this neuron
        return weights
    
    def apply_hip(self, neuron, duration):
        """
        Apply Homeostatic Intrinsic Plasticity (HIP) to adjust the neuron's firing threshold.
        
        Parameters:
        - neuron (LIFNeuron): The neuron to adjust.
        - duration (float): Duration over which the neuron's activity is measured in ms.
        """
        if self.use_hip:
            actual_spike_rate = neuron.spike_count / duration
            print(f"Actual spike rate: {actual_spike_rate}, Target spike rate: {self.target_spike_rate}")

            # Save the current v_threshold value
            self.v_threshold_history.append(neuron.v_threshold)

            # Adjust firing threshold based on the difference between actual and target spike rates
            if actual_spike_rate > self.target_spike_rate:
                neuron.v_threshold += self.learning_rate * (actual_spike_rate - self.target_spike_rate)
            else:
                neuron.v_threshold -= self.learning_rate * (self.target_spike_rate - actual_spike_rate)

            # Reset spike count after adjustment
            neuron.spike_count = 0

    def apply_lateral_inhibition(self, neurons, spikes, current_time):
        """
        Apply lateral inhibition to reduce the activity of neighboring neurons.
        
        Parameters:
        - neurons (list): List of LIFNeuron objects.
        - spikes (ndarray): Array indicating which neurons spiked.
        - current_time (float): Current simulation time in ms.
        """
        if self.use_lateral_inhibition:
            num_neurons = len(neurons)
            inhibition_values = {1: 2.0, 2: 1.0, 3: 0.5, 4: 0.25, 5: 0.125}  # Define inhibition values for neighbors
            
            for i, spiked in enumerate(spikes):
                if spiked:  # If the neuron fired
                    for distance, inhibition_value in inhibition_values.items():
                        neighbors = [(i - distance) % num_neurons, (i + distance) % num_neurons]
                        for neighbor in neighbors:
                            neurons[neighbor].v_m -= inhibition_value * self.lateral_inhibition_strength
                            if neighbor not in self.inhibited_neurons:
                                self.inhibited_neurons[neighbor] = []
                            self.inhibited_neurons[neighbor].append(current_time)  # Record inhibition time

    def apply_global_inhibition(self, weights, spikes):
        """
        Apply global inhibition by scaling down synaptic weights if the total spike count exceeds a threshold.
        
        Parameters:
        - weights (ndarray): Synaptic weights matrix.
        - spikes (ndarray): Array indicating which neurons spiked.
        
        Returns:
        - ndarray: Scaled weights if global inhibition is applied, otherwise original weights.
        """
        total_spikes = np.sum(spikes)
        if self.use_global_inhibition and total_spikes > self.global_inhibition_threshold:
            weights *= self.global_inhibition_scale  # Scale all synaptic weights
        return weights

    def plot_v_threshold(self):
        """
        Plot the changes in firing threshold (v_threshold) over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.v_threshold_history, label='v_threshold')
        plt.xlabel('HIP Adjustment Steps')
        plt.ylabel('v_threshold (mV)')
        plt.title('Change in Firing Threshold (v_threshold) Over Time')
        plt.legend()
        plt.show()

    def plot_raster_with_inhibition(self, spike_times, duration):
        """
        Plot a raster plot showing neuron spike times and lateral inhibition events.
        
        Parameters:
        - spike_times (list): List of spike times for each neuron.
        - duration (float): Duration of the simulation in ms.
        """
        plt.figure(figsize=(12, 8))
        
        for i in range(len(spike_times)):
            plt.scatter(spike_times[i], [i] * len(spike_times[i]), color='black', s=2)

        # Draw inhibited neurons
        for neuron_idx, times in self.inhibited_neurons.items():
            plt.scatter(times, [neuron_idx] * len(times), color='red', s=2)

        plt.title('Raster Plot with Lateral Inhibition')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.show()
