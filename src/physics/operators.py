import torch
import math

def solubility(T, a0, a1, a2):
    return a0 + a1 * T + a2 * T * T

def supersaturation(c_phys, T_phys, a0, a1, a2):
    c_s = solubility(T_phys, a0, a1, a2)
    ratio = c_phys / c_s
    return torch.where(c_phys >= c_s, ratio - 1.0, 1.0 - ratio)

def nucleation_rate(sigma, c_phys, T_phys, kb, b, a0, a1, a2):
    c_s = solubility(T_phys, a0, a1, a2)
    cond = (c_phys >= c_s)
    return torch.where(cond, kb * torch.abs(sigma) ** b, torch.zeros_like(sigma))

def growth_rate(sigma, c_phys, T_phys, kg, g, kd, d, a0, a1, a2):
    c_s = solubility(T_phys, a0, a1, a2)
    cond = (c_phys >= c_s)
    growth = kg * torch.abs(sigma) ** g
    dissolution = -kd * torch.abs(sigma) ** d
    return torch.where(cond, growth, dissolution)

def selection_function(lamb_phys, N, rho_imp, h_imp, l_imp, d_imp, beta):
    prefactor = (rho_imp * h_imp * l_imp * (d_imp ** 2)) / (768.0 * math.pi)
    return prefactor * (N ** 3) * (lamb_phys ** beta)