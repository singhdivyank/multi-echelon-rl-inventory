"""
Utility functions for neural networks
"""
import torch
import torch.nn as nn

def init_weights(module, gain=1.0):
    """
    Initialize network weights using Glorot-Uniform (Xavier uniform)
    As specified in Table 4 of the paper
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def get_activation(activation_name: str):
    """
    Get activation function by name
    
    Args:
        activation_name: Name of activation ('tanh', 'relu', 'elu', etc.)
        
    Returns:
        Activation function
    """
    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'leaky_relu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
    }
    
    if activation_name.lower() not in activations:
        raise ValueError(f"Unknown activation: {activation_name}")
        
    return activations[activation_name.lower()]


def count_parameters(model):
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get available device (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target_net, source_net, tau):
    """
    Soft update target network parameters
    θ_target = τ * θ_source + (1 - τ) * θ_target
    """
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data
        )


def hard_update(target_net, source_net):
    """Hard update target network parameters"""
    target_net.load_state_dict(source_net.state_dict())