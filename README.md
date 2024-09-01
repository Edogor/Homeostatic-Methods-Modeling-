# Spiking Neural Networks with Homeostatic Mechanisms

This project implements various homeostatic mechanisms in a spiking neural network (SNN) to maintain network stability and prevent weight overgrowth.

## Homeostatic Mechanisms

- **Weight Normalization**: Ensures synaptic weights remain balanced.
- **Synaptic Scaling**: Adjusts synaptic strengths to maintain stability.
- **Homeostatic Intrinsic Plasticity (HIP)**: Dynamically adjusts neuron firing thresholds.
- **Lateral Inhibition**: Suppresses neighboring neurons to prevent runaway excitation.
- **Global Inhibition**: Scales down network-wide synaptic strengths when activity exceeds a threshold.

## Usage
In the network.py you can acitvate or deactive the homeostatic function by setting them to True or False.

### Installation

To run the project, clone this repository and install the required dependencies:

### Installation

```bash
git clone https://github.com/Edogor/Homeostatic-Methods-Modeling-.git
cd Homeostatic-Methods-Modeling
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

```
### execution
```bash
python network.py
