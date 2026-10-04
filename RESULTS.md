# TLDR

I took a 5000 neuron feeding circuit from the FlyWire connectome, ported a spiking teacher to GPU, trained MORGAN-style spectral MoE models and compared them to baselines and controls.

**Main stuff:** the graph models lost badly to a graph-free MLP... the MoE router collapsed to one expert... 3/4 experts were inert... connectome structure mattered versus a degree/sign-matched rewiring... direction did not help.

NOT evidence that fly brains use MORGAN-style spectral MoE. I guess its not rigorous enough but points to evidence that this particular task did not engage the architecture.

THX for reading ;P
