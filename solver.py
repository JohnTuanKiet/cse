"""
RF propagation 
"""

import numpy as np
import matplotlib.pyplot as plt

def advanced_vectorized(u: np.ndarray, u_1: np.ndarray, u_2: np.ndarray, c: float, dt: float, dx: float, dy: float, f: function, 
                        I: function, V: function, step1: bool = False, bc: str = "von-neumann") -> np.ndarray:
    """
    Purpose: Step forward in time with explicit scheme
    
    Input: \n
    u -- system at n-1 time step \n
    u_1 -- system at n time step \n
    c -- propagation constant \n
    step -- 
    dt -- 
    dx -- 
    dy --
    f -- external factor \n
    I -- intital state of the system \n
    V -- initial derivative of the system \n
    boundary_condition -- setting a boundary condition for the system \n
    step1 -- handle different for first step of the system \n
    
    Output: \n
    u_2 -- System at n+1 time step
    """
    
    dt2 = dt ** 2
    c2 = c ** 2
    c_x = c2/ ((dx ** 2) * (dt2))
    c_y = c2 / ((dy ** 2) * (dt2))
    
    u_1_mid = u_1[1:-1, 1:-1]
    u_1_left = u_1[2:, 1:-1]
    u_1_right = u_1[:-2, 1:-1]
    u_1_top = u_1[1:-1, 2:]
    u_1_bottom = + u_1[1:-1, :-2]
    
    u_xx = u_1_right - 2 * u_1_mid + u_1_left
    u_yy = u_1_top - 2 * u_1_mid + u_1_bottom
    
    # Handle step 1
    if step1: 
        u_2[1:-1, 1:-1] = 1/2 * c_x * u_xx + 1/2 * c_y * u_yy + 1/2 * f[1:-1, 1:-1] * dt2 + u_1[1:-1, 1:-1] + V[1:-1, 1:-1] * dt
        
    else:
        u_2[1:-1, 1:-1] = c_x * u_xx + c_y * u_yy + f[1:-1, 1:-1] * dt2 + 2 * u_1[1:-1] - u
        
    if bc == "von-neumann":
        if step1:
            u_2[0, 1:-1] = c_x * 2 * (u_1[1, 1:-1] - u_1[0, 1:-1]) + c_y * (u[0, 2:] - u[0, 1:-1] + u[0, :-2]) + \
            f[0, 1:-1] * dt2 + u_1[0, 1:-1] + V[0, 1:-1] * dt
            u_2[-1, 1:-1] = c_x * 2 * (u_1[-2, 1:-1] - u_1[-1, 1:-1]) + c_y * (u[-1, 2:] - u[-1, 1:-1] + u[-1, :-2]) + \
            f[-1, 1:-1] * dt2 + u_1[-1, 1:-1] + V[-1, 1:-1] * dt
            u_2[1:-1, 0] = c_x * (u[2:, 0] - u[1:-1, 0] + u[:-2, 0]) + c_y * 2 * (u_1[1:-1, 1] - u_1[1:-1, 0]) + \
            f[1:-1, 0] * dt2 + u_1[1:-1, 0] + V[1:-1, 0] * dt
            u_2[1:-1, -1] = c_x * (u[2:, -1] - u[1:-1, -1] + u[:-2, -1]) + c_y * 2 * (u_1[1:-1, -2] - u_1[1:-1, -1]) + \
            f[1:-1, -1] * dt2 + u_1[1:-1, -1] + V[1:-1, -1] * dt
            u_2[0, 0] = c_x * 2 * (u_1[1, 0] - u_1[0, 0]) + c_y * 2 * (u[0, 1] - u[0, 0]) + \
            f[0, 0] * dt2 + u_1[0, 0] + V[0, 0] * dt
            u_2[0, -1] = c_x * 2 * (u_1[1, -1] - u_1[0, -1]) + c_y * 2 * (u[0, -2] - u[0, -1]) + \
            f[0, -1] * dt2 + u_1[0, -1] + V[0, -1] * dt
            u_2[-1, 0] = c_x * 2 * (u_1[-2, 0] - u_1[-1, 0]) + c_y * 2 * (u[-1, 1] - u[-1, 0]) + \
            f[-1, 0] * dt2 + u_1[-1, 0] + V[-1, 0] * dt
            u_2[-1, -1] = c_x * 2 * (u_1[-2, -1] - u_1[-1, -1]) + c_y * 2 * (u[-1, -2] - u[-1, -1]) + \
            f[-1, -1] * dt2 + u_1[-1, -1] + V[-1, -1] * dt
        else: 
            u_2[0, 1:-1] = c_x * 2 * (u_1[1, 1:-1] - u_1[0, 1:-1]) + c_y * (u[0, 2:] - u[0, 1:-1] + u[0, :-2]) + \
            f[0, 1:-1] * dt2 + 2 * u_1[0, 1:-1] - u[0, 1:-1]
            u_2[-1, 1:-1] = c_x * 2 * (u_1[-2, 1:-1] - u_1[-1, 1:-1]) + c_y * (u[-1, 2:] - u[-1, 1:-1] + u[-1, :-2]) + \
            f[-1, 1:-1] * dt2 + 2 * u_1[-1, 1:-1] - u[-1, 1:-1]
            u_2[1:-1, 0] = c_x * (u[2:, 0] - u[1:-1, 0] + u[:-2, 0]) + c_y * 2 * (u_1[1:-1, 1] - u_1[1:-1, 0]) + \
            f[1:-1, 0] * dt2 + 2 * u_1[1:-1, 0] - u[1:-1, 0]
            u_2[1:-1, -1] = c_x * (u[2:, -1] - u[1:-1, -1] + u[:-2, -1]) + c_y * 2 * (u_1[1:-1, -2] - u_1[1:-1, -1]) + \
            f[1:-1, -1] * dt2 + 2 * u_1[1:-1, -1] - u[1:-1, -1]
            u_2[0, 0] = c_x * 2 * (u_1[1, 0] - u_1[0, 0]) + c_y * 2 * (u[0, 1] - u[0, 0]) + \
            f[0, 0] * dt2 + 2 * u_1[0, 0] - u[0, 0]
            u_2[0, -1] = c_x * 2 * (u_1[1, -1] - u_1[0, -1]) + c_y * 2 * (u[0, -2] - u[0, -1]) + \
            f[0, -1] * dt2 + 2 * u_1[0, -1] - u[0, -1]
            u_2[-1, 0] = c_x * 2 * (u_1[-2, 0] - u_1[-1, 0]) + c_y * 2 * (u[-1, 1] - u[-1, 0]) + \
            f[-1, 0] * dt2 + 2 * u_1[-1, 0] - u[-1, 0]
            u_2[-1, -1] = c_x * 2 * (u_1[-2, -1] - u_1[-1, -1]) + c_y * 2 * (u[-1, -2] - u[-1, -1]) + \
            f[-1, -1] * dt2 + 2 * u_1[-1, -1] - u[-1, -1]
            

def solver(c: float, dt: float, dx: float, dy: float, f: function, 
           I: function, V: function, step1: bool = False, bc: str = "von-neumann") -> np.ndarray:
    
    return 0


def main():
    return 0