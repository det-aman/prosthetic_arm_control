# In snn_model.py

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Define the SNN architecture
class SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta=0.95):
        super().__init__()

        # Define the surrogate gradient for backpropagation
        spike_grad = surrogate.fast_sigmoid()

        # Layer 1: Input to Hidden Layer
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Layer 2: Hidden to Output Layer
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer's membrane potential for classification
        output_potentials = []

        # Process data over time steps (the "spiking" part)
        for step in range(x.size(1)):  # x.size(1) is the number of time steps
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            output_potentials.append(mem2)

        # Stack and sum the potentials over time
        return torch.stack(output_potentials, dim=1).sum(dim=1)