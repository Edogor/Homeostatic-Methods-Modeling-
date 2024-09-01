import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from lif_neuron import LIFNeuron
from stdp import STDP
from homeostatic_mechanisms import HomeostaticMechanisms

def compare_networks(network_with_homeostasis, network_without_homeostasis, input_spike_trains, duration):
    """
    Compare two networks: one with homeostatic mechanisms and one without.

    Parameters:
    - network_with_homeostasis (NeuralNetwork): Network with homeostatic mechanisms enabled.
    - network_without_homeostasis (NeuralNetwork): Network without homeostatic mechanisms.
    - input_spike_trains (list): List of input spike trains for each neuron.
    - duration (float): Duration of the simulation in ms.
    """
    # Simulate both networks
    spike_times_with_homeostasis = network_with_homeostasis.simulate(duration=duration, dt=1, input_spike_trains=input_spike_trains)
    spike_times_without_homeostasis = network_without_homeostasis.simulate(duration=duration, dt=1, input_spike_trains=input_spike_trains)
    
    # Get initial and final weights for both networks
    initial_weights_with_homeostasis = network_with_homeostasis.initial_weights
    final_weights_with_homeostasis = network_with_homeostasis.weights
    
    initial_weights_without_homeostasis = network_without_homeostasis.initial_weights
    final_weights_without_homeostasis = network_without_homeostasis.weights
    
    # Capture active homeostatic mechanisms for the network with homeostasis
    active_mechanisms = []
    if network_with_homeostasis.homeostasis.use_weight_normalization:
        active_mechanisms.append("Weight Normalization")
    if network_with_homeostasis.homeostasis.use_synaptic_scaling:
        active_mechanisms.append("Synaptic Scaling")
    if network_with_homeostasis.homeostasis.use_hip:
        active_mechanisms.append("Homeostatic Intrinsic Plasticity (HIP)")
    if network_with_homeostasis.homeostasis.use_lateral_inhibition:
        active_mechanisms.append("Lateral Inhibition")
    if network_with_homeostasis.homeostasis.use_global_inhibition:
        active_mechanisms.append("Global Inhibition")

    mechanisms_text = "\n".join(active_mechanisms) if active_mechanisms else "None"

    # Plotting
    plt.figure(figsize=(16, 12))

    # Active Homeostatic Mechanisms (on the left, above Network 1)
    plt.subplot(3, 2, 1)
    plt.text(0.5, 0.5, f"Active Homeostatic Mechanisms:\n{mechanisms_text}",
             fontsize=10, ha='center', va='center')
    plt.axis('off')  # Hide the axis

    # Input Spike Trains (on the right, above Network 2)
    plt.subplot(3, 2, 2)
    for neuron_idx, spike_train in enumerate(input_spike_trains):
        spike_times_input = np.where(spike_train == 1)[0] * (duration / len(spike_train))
        plt.scatter(spike_times_input, [neuron_idx] * len(spike_times_input), color='blue', s=2)
    plt.title("Input Spike Trains (Poisson-distributed)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Input Neuron Index")
    
    # Raster Plot for Network with Homeostasis
    plt.subplot(3, 2, 3)
    for neuron_idx, spikes in enumerate(spike_times_with_homeostasis):
        plt.scatter(spikes, [neuron_idx] * len(spikes), color='black', s=2)
    plt.title("Raster Plot - Network 1 (With Homeostasis)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index")
    
    # Raster Plot for Network without Homeostasis
    plt.subplot(3, 2, 4)
    for neuron_idx, spikes in enumerate(spike_times_without_homeostasis):
        plt.scatter(spikes, [neuron_idx] * len(spikes), color='black', s=2)
    plt.title("Raster Plot - Network 2 (Without Homeostasis)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index")
    
    # Plot Initial and Final Weights for Network with Homeostasis
    plt.subplot(3, 2, 5)
    plt.plot(initial_weights_with_homeostasis.flatten(), 'bo', label="Initial Weights")
    plt.plot(final_weights_with_homeostasis.flatten(), 'ro', label="Final Weights")
    plt.title("Synaptic Weights - Network 1")
    plt.xlabel("Synapse Index")
    plt.ylabel("Weight Value")
    plt.legend()
    
    # Plot Initial and Final Weights for Network without Homeostasis
    plt.subplot(3, 2, 6)
    plt.plot(initial_weights_without_homeostasis.flatten(), 'bo', label="Initial Weights")
    plt.plot(final_weights_without_homeostasis.flatten(), 'ro', label="Final Weights")
    plt.title("Synaptic Weights - Network 2")
    plt.xlabel("Synapse Index")
    plt.ylabel("Weight Value")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

class NeuralNetwork:
    def __init__(self, num_neurons, connection_prob=0.8, use_weight_normalization=False, use_synaptic_scaling=False, target_synaptic_sum=5.0, use_hip=False, use_lateral_inhibition=False, use_global_inhibition=False):
        """
        Initialize a neural network with specified homeostatic mechanisms.
        
        Parameters:
        - num_neurons (int): Number of neurons in the network.
        - connection_prob (float): Probability of connection between neurons.
        - use_weight_normalization (bool): Enable weight normalization.
        - use_synaptic_scaling (bool): Enable synaptic scaling.
        - target_synaptic_sum (float): Target sum of synaptic inputs for scaling.
        - use_hip (bool): Enable Homeostatic Intrinsic Plasticity (HIP).
        - use_lateral_inhibition (bool): Enable lateral inhibition.
        - use_global_inhibition (bool): Enable global inhibition.
        """
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
        self.stdp = STDP()
        self.weights = np.random.uniform(0.1, 1, (num_neurons, num_neurons)) * (np.random.rand(num_neurons, num_neurons) < connection_prob)
        self.initial_weights = self.weights.copy()
        self.homeostasis = HomeostaticMechanisms(use_weight_normalization=use_weight_normalization, use_synaptic_scaling=use_synaptic_scaling, target_synaptic_sum=target_synaptic_sum, use_hip=use_hip, use_lateral_inhibition=use_lateral_inhibition, use_global_inhibition=use_global_inhibition)

    def simulate(self, duration, dt, input_spike_trains):
        """
        Simulate the neural network over a given duration with a specified time step.
        
        Parameters:
        - duration (float): Duration of the simulation in ms.
        - dt (float): Time step for the simulation in ms.
        - input_spike_trains (list): List of input spike trains for each neuron.
        
        Returns:
        - list: Spike times for each neuron.
        """
        num_steps = int(duration / dt)
        spike_times = [[] for _ in range(len(self.neurons))]
        spikes = np.zeros(self.num_neurons)

        for t in range(num_steps):
            current_time = t * dt
            new_spikes = np.zeros(self.num_neurons)

            for i, neuron in enumerate(self.neurons):
                external_input = input_spike_trains[i][t]
                synaptic_input = np.dot(self.weights[:, i], spikes)
                total_synaptic_input = external_input + synaptic_input / 8.0   # Scale synaptic input
                
                # Update the neuron and check if it spikes
                spiked = neuron.update(total_synaptic_input, dt)

                if spiked:
                    new_spikes[i] = 1
                    spike_times[i].append(current_time)

            # Apply lateral inhibition (if activated)
            self.homeostasis.apply_lateral_inhibition(self.neurons, new_spikes, current_time)

            # Apply global inhibition if necessary (if activated)
            self.weights = self.homeostasis.apply_global_inhibition(self.weights, new_spikes)

            # Update synaptic weights based on STDP
            for i, neuron in enumerate(self.neurons):
                if new_spikes[i]:
                    for j in range(self.num_neurons):
                        if self.weights[j, i] > 0:
                            delta_t = current_time - spike_times[j][-1] if spike_times[j] else np.inf
                            self.weights[j, i] += self.stdp.update_weight(delta_t)
                            self.weights[j, i] = max(0, self.weights[j, i])

            spikes = new_spikes

            # Apply weight normalization and synaptic scaling (if activated)
            self.weights = self.homeostasis.apply_weight_normalization(self.weights)
            self.weights = self.homeostasis.apply_synaptic_scaling(self.weights)

            # Apply Homeostatic Intrinsic Plasticity (HIP) every 30 ms (if activated)
            hip_frequency = 30
            if self.homeostasis.use_hip and t % hip_frequency == 0:
                for neuron in self.neurons:
                    self.homeostasis.apply_hip(neuron, dt * hip_frequency)

        return spike_times

    def generate_poisson_spike_train(self, rate, duration, dt):
        """
        Generate a Poisson-distributed spike train.

        Parameters:
        - rate (float): Firing rate in spikes per second.
        - duration (float): Duration of the spike train in ms.
        - dt (float): Time step in ms.

        Returns:
        - ndarray: Generated spike train.
        """
        num_steps = int(duration / dt)
        spike_train = np.random.rand(num_steps) < rate * dt / 1000.0
        return spike_train.astype(int)

    def plot_results(self, spike_times, input_spike_trains):
        """
        Plot the input spike trains, raster plot of neuron spikes, and synaptic weights.
        
        Parameters:
        - spike_times (list): List of spike times for each neuron.
        - input_spike_trains (list): List of input spike trains for each neuron.
        """
        num_neurons = len(spike_times)

        plt.figure(figsize=(12, 10))

        # Plot Input Spike Trains
        plt.subplot(3, 1, 1)
        for neuron_idx, spike_train in enumerate(input_spike_trains):
            spike_times_input = np.where(spike_train == 1)[0] * (1000 / len(spike_train))
            plt.scatter(spike_times_input, [neuron_idx] * len(spike_times_input), color='blue', s=2)
        plt.title("Input Spike Trains (Poisson-distributed)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Input Neuron Index")

        # Plot Raster Plot (Spike Times)
        plt.subplot(3, 1, 2)
        for neuron_idx, spikes in enumerate(spike_times):
            plt.scatter(spikes, [neuron_idx] * len(spikes), color='black', s=2)
        plt.title("Raster Plot (Spike Times)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")

        # Plot Initial and Final Weights
        plt.subplot(3, 1, 3)
        plt.plot(self.initial_weights.flatten(), 'bo', label="Initial Weights")
        plt.plot(self.weights.flatten(), 'ro', label="Final Weights")
        plt.title("Synaptic Weights Before and After Simulation")
        plt.xlabel("Synapse Index")
        plt.ylabel("Weight Value")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_network(self):
        """
        Plot the neural network architecture using NetworkX and Matplotlib.
        """
        G = nx.DiGraph()

        # Add nodes
        for i in range(self.num_neurons):
            G.add_node(i)

        # Add edges with weights as attributes
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if self.weights[i, j] > 0:
                    G.add_edge(i, j, weight=self.weights[i, j])

        pos = nx.circular_layout(G)  # Circular layout for visualization
        edge_labels = nx.get_edge_attributes(G, 'weight')

        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=12, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Neural Network Architecture")
        plt.show()

# Example usage
if __name__ == "__main__":
    num_neurons = 30
    duration = 6000 # 6000 ms simulation
    dt = 1  # 1 ms time step

    # Generate the same Poisson-distributed inputs for both networks
    input_spike_trains = [NeuralNetwork(num_neurons).generate_poisson_spike_train(rate=60, duration=duration, dt=dt) for _ in range(num_neurons)]

    # Network 1 with homeostatic mechanisms
    network_with_homeostasis = NeuralNetwork(num_neurons=num_neurons, use_weight_normalization=False, use_synaptic_scaling=False, use_hip=False, use_lateral_inhibition=False, use_global_inhibition=True)
    spike_times = network_with_homeostasis.simulate(duration, dt, input_spike_trains)

    # Network 2 without homeostatic mechanisms
    network_without_homeostasis = NeuralNetwork(num_neurons=num_neurons, use_weight_normalization=False, use_synaptic_scaling=False, use_hip=False, use_lateral_inhibition=False, use_global_inhibition=False)
    network_without_homeostasis.simulate(duration, dt, input_spike_trains)

    # Plot comparison
    compare_networks(network_with_homeostasis, network_without_homeostasis, input_spike_trains, duration=duration)

    # Optional: Uncomment to visualize specific results
    # network_with_homeostasis.homeostasis.plot_raster_with_inhibition(spike_times, duration)
    # network_with_homeostasis.plot_network()
    # network_with_homeostasis.homeostasis.plot_v_threshold()
