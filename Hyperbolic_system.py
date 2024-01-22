from calendar import c
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse.linalg import lsqr
from scipy.sparse import csc_matrix
import pdb
from matplotlib import cm
import math
from sklearn.model_selection import train_test_split
from functools import reduce
from functools import partial
import operator
from timeit import default_timer
from matplotlib.ticker import FormatStrFormatter
import deepxde as dde
import time
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from timeit import default_timer
from matplotlib.ticker import FormatStrFormatter

def solver_2x2(params, N_g):
    Delta = 1 / (N_g - 1)
    numTot = N_g * (N_g + 1) // 2

    # Create index maps for storing values
    index_map = np.zeros((N_g, N_g), dtype=int)
    counter = 1
    for s in range(0,N_g):
        for i in range(0,s+1):
            index_map[i, s] = counter
            counter += 1

    indexes = np.tile(np.arange(1, 2*(numTot+1)-1)[:, np.newaxis], (1, 5))
    weights = np.zeros((2 * numTot, 5))
    RHS = np.zeros(2 * numTot)

    # Directional derivatives
    s_K_x, s_K_y = np.meshgrid(params['mu'], -params['lambda'])
    s_L_x, s_L_y = np.meshgrid(params['mu'], params['mu'])

    s_K = np.sqrt(s_K_x**2 + s_K_y**2)
    s_L = np.sqrt(s_L_x**2 + s_L_y**2)

    # Iterate over triangular domain for K
    for indX in range(0, N_g):
        for indY in range(0, indX+1):
            glob_ind = index_map[indY, indX]-1
            handle_K(s_K,indX, indY, glob_ind, s_K_x, s_K_y, Delta, params, weights, indexes, RHS, index_map, numTot)

    # Iterate over triangular domain for L
    for indX in range(0, N_g):
        for indY in range(0, indX+1):
            glob_ind = index_map[indY, indX]-1 + numTot
            handle_L(s_L,indX, indY, glob_ind, s_L_x, s_L_y, Delta, params, weights, indexes, RHS, index_map, numTot)

    # Source terms
    for indX in range(0,N_g):
        for indY in range(0, indX+1):
            glob_ind = index_map[indY, indX]-1

            if indY != indX:
                weights[glob_ind, 3] = -params['a_1'][indY, indX]
                weights[glob_ind, 4] = -params['a_2'][indY, indX]
                indexes[glob_ind, 4] = glob_ind + numTot+1

            if indY != 0:
                glob_ind_L = glob_ind + numTot
                weights[glob_ind_L, 3] = -params['b_1'][indY, indX]
                indexes[glob_ind_L, 3] = glob_ind+1
                weights[glob_ind_L, 4] = -params['b_2'][indY, indX]

    # Assemble sparse matrix A and vector b
    A_vals = weights.ravel()
    A_rows = np.repeat(np.arange(1,2 * numTot+1), 5)
    A_cols = indexes.ravel()
    b_vals = RHS

    A = csc_matrix((A_vals.flatten('F'),
                (A_rows.astype(int).flatten('F')-1,
                 A_cols.astype(int).flatten('F')-1)),shape=(2 * numTot,2 * numTot))

    b = csc_matrix((b_vals.flatten('F'),
                 (np.arange(2 * numTot).astype(int).flatten('F'),
                  np.zeros(2 * numTot).astype(int).flatten('F'))), shape=(2 * numTot, 1))

    # Solve the linear system
    x = spsolve(A, b)
    K = np.zeros((N_g, N_g))
    L = np.zeros((N_g, N_g))

    counter =0# 初始化计数器
    for s in range(0,N_g):
        for i in range(0,s+1):
            K[i, s] = x[counter]
            L[i, s] = x[counter+numTot]
            counter += 1
    return K, L


def solver_2x2_observer(params, N_g):
    Delta = 1 / (N_g - 1)
    numTot = N_g * (N_g + 1) // 2

    # Create index maps for storing values
    index_map = np.zeros((N_g, N_g), dtype=int)
    counter = 1
    for s in range(0,N_g):
        for i in range(0,s+1):
            index_map[i, s] = counter
            counter += 1

    indexes = np.tile(np.arange(1, 2*(numTot+1)-1)[:, np.newaxis], (1, 5))
    weights = np.zeros((2 * numTot, 5))
    RHS = np.zeros(2 * numTot)

    # Directional derivatives
    s_K_x, s_K_y = np.meshgrid(params['lambda'], -params['mu'])
    s_L_x, s_L_y = np.meshgrid(params['mu'], params['mu'])

    s_K = np.sqrt(s_K_x**2 + s_K_y**2)
    s_L = np.sqrt(s_L_x**2 + s_L_y**2)

    # Iterate over triangular domain for K
    for indX in range(0, N_g):
        for indY in range(0, indX+1):
            glob_ind = index_map[indY, indX]-1
            handle_K_observer(s_K,indX, indY, glob_ind, s_K_x, s_K_y, Delta, params, weights, indexes, RHS, index_map, numTot)

    # Iterate over triangular domain for L
    for indX in range(0, N_g):
        for indY in range(0, indX+1):
            glob_ind = index_map[indY, indX]-1 + numTot
            # print(glob_ind)
            handle_L_observer(s_L,indX, indY, glob_ind, s_L_x, s_L_y, Delta, params, weights, indexes, RHS, index_map, numTot)

    for indX in range(0,N_g):
        for indY in range(0, indX+1):
            glob_ind = index_map[indY, indX]-1
            if indY != indX:
                weights[glob_ind, 3] = -params['a_1'][indY, indX]
                weights[glob_ind, 4] = -params['a_2'][indY, indX]
                indexes[glob_ind, 4] = glob_ind + numTot+1
            if indY != 0:
                glob_ind_L = glob_ind + numTot
                weights[glob_ind_L, 3] = -params['b_1'][indY, indX]
                indexes[glob_ind_L, 3] = glob_ind+1
                weights[glob_ind_L, 4] = -params['b_2'][indY, indX]

    A_vals = weights.ravel()
    A_rows = np.repeat(np.arange(1,2 * numTot+1), 5)
    A_cols = indexes.ravel()
    b_vals = RHS

    A = csc_matrix((A_vals.flatten('F'),
                (A_rows.astype(int).flatten('F')-1,
                 A_cols.astype(int).flatten('F')-1)),shape=(2 * numTot,2 * numTot))

    b = csc_matrix((b_vals.flatten('F'),
                 (np.arange(2 * numTot).astype(int).flatten('F'),
                  np.zeros(2 * numTot).astype(int).flatten('F'))), shape=(2 * numTot, 1))

    # Solve the linear system
    x = spsolve(A, b)
    K = np.zeros((N_g, N_g))
    L = np.zeros((N_g, N_g))

    counter =0
    for s in range(0,N_g):
        for i in range(0,s+1):
            K[i, s] = x[counter]
            L[i, s] = x[counter+numTot]
            counter += 1
    return K, L

def handle_K_observer(s_K, indX, indY, glob_ind, s_K_x, s_K_y, Delta, params, weights, indexes, RHS, index_map, numTot):
    if indY == indX:  # Boundary
        weights[glob_ind , 0] = 1
        RHS[glob_ind] = params['f'][indY ]
    else:
        theta = np.arctan2(s_K_x[indY, indX], -s_K_y[indY, indX])
        if theta > np.pi / 4:  # Left side
            sigma = Delta / np.sin(theta)
            d = sigma * np.cos(theta)
            preFactor = s_K[indY, indX] / sigma
            weights[glob_ind, 0] = preFactor
            weights[glob_ind, 1] = -preFactor * d / Delta
            indexes[glob_ind, 1] = index_map[indY+1, indX-1]
            weights[glob_ind, 2] = -preFactor * (Delta - d) / Delta
            indexes[glob_ind, 2] = index_map[indY, indX-1]

        else:  # Top
            sigma = Delta / np.cos(theta)
            d = sigma * np.sin(theta)
            preFactor = s_K[indY, indX] / sigma

            weights[glob_ind, 0] = preFactor
            weights[glob_ind, 1] = -preFactor * d / Delta
            weights[glob_ind, 2] = -preFactor * (Delta - d) / Delta
            indexes[glob_ind, 1] = index_map[indY+1, indX-1]
            indexes[glob_ind, 2] = index_map[indY+1, indX]

        if indY == indX - 1:  # Subdiagonal
            if theta > np.pi / 4:  # Left side
                weights[glob_ind, 2] += weights[glob_ind, 1]
                weights[glob_ind, 0] -= weights[glob_ind, 1]
                indexes[glob_ind, 1] = index_map[indY+1, indX]
            else:  # Top
                weights[glob_ind, 2] += weights[glob_ind, 1]
                weights[glob_ind, 0] -= weights[glob_ind, 1]
                indexes[glob_ind, 1] = index_map[indY, indX - 1]



def handle_K(s_K, indX, indY, glob_ind, s_K_x, s_K_y, Delta, params, weights, indexes, RHS, index_map, numTot):
    if indY == indX:  # Boundary
        weights[glob_ind , 0] = 1
        RHS[glob_ind] = params['f'][indY ]
    else:
        theta = np.arctan2(s_K_x[indY, indX], -s_K_y[indY, indX])
        if theta > np.pi / 4:  # Left side
            sigma = Delta / np.sin(theta)
            d = sigma * np.cos(theta)
            preFactor = s_K[indY, indX] / sigma
            weights[glob_ind, 0] = preFactor
            weights[glob_ind, 1] = -preFactor * d / Delta
            indexes[glob_ind, 1] = index_map[indY+1, indX-1]
            weights[glob_ind, 2] = -preFactor * (Delta - d) / Delta
            indexes[glob_ind, 2] = index_map[indY, indX-1]

        else:  # Top
            sigma = Delta / np.cos(theta)
            d = sigma * np.sin(theta)
            preFactor = s_K[indY, indX] / sigma

            weights[glob_ind, 0] = preFactor
            weights[glob_ind, 1] = -preFactor * d / Delta
            weights[glob_ind, 2] = -preFactor * (Delta - d) / Delta
            indexes[glob_ind, 1] = index_map[indY, indX]
            indexes[glob_ind, 2] = index_map[indY, indX]

        if indY == indX - 1:  # Subdiagonal
            if theta > np.pi / 4:  # Left side
                weights[glob_ind, 2] += weights[glob_ind, 1]
                weights[glob_ind, 0] -= weights[glob_ind, 1]
                indexes[glob_ind, 1] = index_map[indY+1, indX]
            else:  # Top
                weights[glob_ind, 2] += weights[glob_ind, 1]
                weights[glob_ind, 0] -= weights[glob_ind, 1]
                indexes[glob_ind, 1] = index_map[indY, indX - 1]



def handle_L(s_L,indX, indY, glob_ind, s_L_x, s_L_y, Delta, params, weights, indexes, RHS, index_map, numTot):
    if indY == 0:  # Boundary
        weights[glob_ind, 0] = -1
        weights[glob_ind, 1] = params['q']
        indexes[glob_ind, 1] = glob_ind - numTot+1

    else:
        theta = np.arctan2(s_L_y[indY, indX], s_L_x[indY, indX])
        if theta < np.pi / 4:  # Left side
            sigma = Delta / np.cos(theta)
            d = sigma * np.sin(theta)
            preFactor = s_L[indY, indX] / sigma

            weights[glob_ind, 0] = preFactor
            weights[glob_ind, 1] = -preFactor * (Delta - d) / Delta
            weights[glob_ind, 2] = -preFactor * d / Delta
            indexes[glob_ind, 1] = index_map[indY, indX-1] + numTot
            indexes[glob_ind, 2] = index_map[indY - 1, indX-1] + numTot
        else:  # Bottom
            sigma = Delta / np.sin(theta)
            d = sigma * np.cos(theta)
            preFactor = s_L[indY, indX] / sigma

            weights[glob_ind, 0] = preFactor
            weights[glob_ind, 1] = -preFactor * d / Delta
            weights[glob_ind, 2] = -preFactor * (Delta - d) / Delta

            indexes[glob_ind, 1] = index_map[indY - 1, indX - 1] + numTot
            indexes[glob_ind, 2] = index_map[indY-1, indX] + numTot

def handle_L_observer(s_L,indX, indY, glob_ind, s_L_x, s_L_y, Delta, params, weights, indexes, RHS, index_map, numTot):
    if indY == 0:  # Boundary
        weights[glob_ind, 0] = -1
        weights[glob_ind, 1] = 0
        indexes[glob_ind, 1] = glob_ind - numTot+1

    else:
        theta = np.arctan2(s_L_y[indY, indX], s_L_x[indY, indX])
        if theta < np.pi / 4:  # Left side
            sigma = Delta / np.cos(theta)
            d = sigma * np.sin(theta)
            preFactor = s_L[indY, indX] / sigma

            weights[glob_ind, 0] = preFactor
            weights[glob_ind, 1] = -preFactor * (Delta - d) / Delta
            weights[glob_ind, 2] = -preFactor * d / Delta
            indexes[glob_ind, 1] = index_map[indY, indX-1] + numTot
            indexes[glob_ind, 2] = index_map[indY - 1, indX-1] + numTot
        else:  # Bottom
            sigma = Delta / np.sin(theta)
            d = sigma * np.cos(theta)
            preFactor = s_L[indY, indX] / sigma

            weights[glob_ind, 0] = preFactor
            weights[glob_ind, 1] = -preFactor * d / Delta
            weights[glob_ind, 2] = -preFactor * (Delta - d) / Delta

            indexes[glob_ind, 1] = index_map[indY - 1, indX - 1] + numTot
            indexes[glob_ind, 2] = index_map[indY-1, indX] + numTot


def K_solver_2x2(fun, N_g):
    xspan = np.linspace(0, 1, N_g)
    params = {}
    params['mu'] = fun['mu'](xspan)
    params['lambda'] = fun['lambda'](xspan)
    params['a_1'] = np.zeros((N_g, N_g))
    params['a_2'] = np.zeros((N_g, N_g))
    params['b_1'] = np.zeros((N_g, N_g))
    params['b_2'] = np.zeros((N_g, N_g))
    for i in range(N_g):
        for j in range(N_g):
            params['a_1'][i, j] = fun['lambda_d'](xspan[j]) + fun['delta'](xspan[j])
            params['a_2'][i, j] = fun['c_2'](xspan[j])
            params['b_1'][i, j] = fun['c_1'](xspan[j])
            params['b_2'][i, j] = -fun['mu_d'](xspan[j])

    
    params['f'] = -fun['c_2'](xspan) / (fun['lambda'](xspan) + fun['mu'](xspan))
    params['q'] = fun['q'] * fun['lambda'](0) / fun['mu'](0)
    Kvu, Kvv = solver_2x2(params, N_g)
    return Kvu, Kvv


def P_solver_2x2(fun, N_g):
    xspan = np.linspace(0, 1, N_g)

    params = {}
    params['mu'] = fun['mu'](xspan)
    params['lambda'] = fun['lambda'](xspan)
    params['a_1'] = np.zeros((N_g, N_g))
    params['a_2'] = np.zeros((N_g, N_g))
    params['b_1'] = np.zeros((N_g, N_g))
    params['b_2'] = np.zeros((N_g, N_g))
    params['f']=np.zeros(N_g)
    for i in range(N_g):
        for j in range(N_g):
            params['a_1'][i, j] = -fun['delta'](xspan[N_g-1-i]) + fun['mu_d'](xspan[N_g-1-j])
            params['a_2'][i, j] = -fun['c_1'](xspan[N_g-1-i])
            params['b_1'][i, j] = fun['c_2'](xspan[N_g-1-i])
            params['b_2'][i, j] = -fun['mu_d'](xspan[N_g-1-j])

        params['mu'][i] = fun['mu'](xspan[N_g - 1 - i])
        params['lambda'][i] = fun['lambda'](xspan[N_g - 1 - i])
        params['f'][i]= fun['c_1'](xspan[N_g  -1 - i]) / (fun['lambda'](xspan[N_g - 1 - i]) + fun['mu'](xspan[N_g - 1 - i]))

    params['q'] = fun['q'] * fun['lambda'](0) / fun['mu'](0)
    Kvu, Kvv = solver_2x2_observer(params, N_g)
    return Kvu, Kvv

def ode_chap_08_06(t, x, sys, model):
    N = sys['N']
    N_grid = sys['N_grid']

    Lambda = sys['lambda']
    mu = sys['mu']
    c_1 = sys['c_1']
    c_2 = sys['c_2']
    DDDelta=sys['Ddelta']

    # Variable extraction and augmentation
    dummy = np.reshape(x[:6*N],(N, 6),order='F')
    dummy_a = np.vstack([2*dummy[0, :] - dummy[1, :], dummy, 2*dummy[-1, :] - dummy[-2, :]])
    u_sf_a = dummy_a[:, 0]
    v_sf_a = dummy_a[:, 1]
    u_of_a = dummy_a[:, 2]
    v_of_a = dummy_a[:, 3]
    u_hat_a = dummy_a[:, 4]
    v_hat_a = dummy_a[:, 5]

    u_sf = u_sf_a[1:sys['N']+1]
    v_sf = v_sf_a[1:sys['N']+1]
    u_of = u_of_a[1:sys['N']+1]
    v_of = v_of_a[1:sys['N']+1]
    u_hat = u_hat_a[1:sys['N']+1]
    v_hat = v_hat_a[1:sys['N']+1]

    U_sf_f = x[6*sys['N']]
    U_of_f = x[6*sys['N']+1]

    dx=1/201
    nx=201
    grid = np.linspace(0, 1, nx+1, dtype=np.float32).reshape(nx+1, 1)
    grid = torch.from_numpy(np.array(grid, dtype=np.float32)).cuda()
    hat_U_input=sys['Kvu1'] * u_hat_a + sys['Kvv1'] * v_hat_a
    absolute_values=np.abs(hat_U_input)
    max_value=np.max(absolute_values)
    hat_U_input = np.array(hat_U_input, dtype=np.float32)
    hatUval = torch.from_numpy(hat_U_input[np.newaxis, :]).cuda()
    hat_U = model((hatUval, grid))
    U_of1 = hat_U.detach().cpu().numpy()
    U_of=U_of1[0][0]

    # # Controller
    Al=np.zeros(nx+1)
    Al[0]=1/3
    Al[-1]=1/3
    for i in range(1,nx):
        if i % 2 == 0:
            Al[i]=4/3
        else:
            Al[i]=2/3
    Al=Al*dx

    if sys['ctrl_on'] == 1:
        U_sf= Al @ (sys['Kvu1'] * u_sf_a)+ Al @ (sys['Kvv1'] * v_sf_a)
    else:
        U_sf = 0

    # System and observer dynamics
    Delta = sys['Delta']
    u_sf_a[0] = sys['q'] * v_sf_a[0]
    v_sf_a[-1] = U_sf
    u_of_a[0] = sys['q'] * v_sf_a[0]
    v_of_a[-1] = U_of
    u_hat_a[0] = sys['q'] * v_sf_a[0]
    v_hat_a[-1] = U_of

    # Spatial derivatives
    u_sf_x = np.concatenate([[(u_sf_a[1] - u_sf_a[0]) / Delta],
                             (u_sf_a[:-3] - 4 * u_sf_a[1:-2] + 3 * u_sf_a[2:-1]) / (2 * Delta)])

    v_sf_x = np.concatenate([(-3 * v_sf_a[1:N_grid-2] + 4 * v_sf_a[2:N_grid-1] - v_sf_a[3:N_grid]) / (2 * Delta),
                             [(v_sf_a[-1] - v_sf_a[-2]) / Delta]])

    u_of_x = np.concatenate([[(u_of_a[1] - u_of_a[0]) / Delta],
                             (u_of_a[:-3] - 4 * u_of_a[1:-2] + 3 * u_of_a[2:-1]) / (2 * Delta)])
    v_of_x = np.concatenate([(-3 * v_of_a[1:N_grid-2] + 4 * v_of_a[2:N_grid-1] - v_of_a[3:N_grid]) / (2 * Delta),
                             [(v_of_a[-1] - v_of_a[-2]) / Delta]])

    u_hat_x = np.concatenate([[(u_hat_a[1] - u_hat_a[0]) / Delta],
                              (u_hat_a[:-3] - 4 * u_hat_a[1:-2] + 3 * u_hat_a[2:-1]) / (2 * Delta)])
    v_hat_x = np.concatenate([(-3 * v_hat_a[1:N_grid-2] + 4 * v_hat_a[2:N_grid-1] - v_hat_a[3:N_grid]) / (2 * Delta),
                             [(v_hat_a[-1] - v_hat_a[-2]) / Delta]])

    u_sf_t = -Lambda[1:N+1] * u_sf_x +DDDelta[1:N+1] * u_sf + c_1[1:N+1] * v_sf

    v_sf_t = mu[1:N+1] * v_sf_x + c_2[1:N+1] * u_sf

    u_of_t = -Lambda[1:N+1] * u_of_x +DDDelta[1:N+1] * u_of+ c_1[1:N+1] * v_of
    v_of_t = mu[1:N+1] * v_of_x + c_2[1:N+1] * u_of

    u_hat_t = -Lambda[1:N+1] * u_hat_x +DDDelta[1:N+1] * u_hat+ c_1[1:N+1] * v_hat + sys['p1'][1:N+1] * (v_of_a[0] - v_hat_a[0])
    v_hat_t = mu[1:N+1] * v_hat_x + c_2[1:N+1] * u_hat + sys['p2'][1:N+1] * (v_of_a[0] - v_hat_a[0])

    U_sf_f_t = sys['gamma_U'] * (U_sf - U_sf_f)
    U_of_f_t = sys['gamma_U'] * (U_of - U_of_f)

    # Parse
    dt = np.concatenate([u_sf_t, v_sf_t, u_of_t, v_of_t, u_hat_t, v_hat_t, [U_sf_f_t, U_of_f_t]])
    return dt 

def ode_chap_08_06(t, x, sys):
    N = sys['N']
    N_grid = sys['N_grid']

    Lambda = sys['lambda']
    mu = sys['mu']
    c_1 = sys['c_1']
    c_2 = sys['c_2']
    DDDelta=sys['Ddelta']

    # Variable extraction and augmentation
    dummy = np.reshape(x[:6*N],(N, 6),order='F')
    dummy_a = np.vstack([2*dummy[0, :] - dummy[1, :], dummy, 2*dummy[-1, :] - dummy[-2, :]])
    u_sf_a = dummy_a[:, 0]
    v_sf_a = dummy_a[:, 1]
    u_of_a = dummy_a[:, 2]
    v_of_a = dummy_a[:, 3]
    u_hat_a = dummy_a[:, 4]
    v_hat_a = dummy_a[:, 5]

    u_sf = u_sf_a[1:sys['N']+1]
    v_sf = v_sf_a[1:sys['N']+1]
    u_of = u_of_a[1:sys['N']+1]
    v_of = v_of_a[1:sys['N']+1]
    u_hat = u_hat_a[1:sys['N']+1]
    v_hat = v_hat_a[1:sys['N']+1]

    U_sf_f = x[6*sys['N']]
    U_of_f = x[6*sys['N']+1]

    dx=1/201
    nx=201
    Al=np.zeros(nx+1)
    Al[0]=1/3
    Al[-1]=1/3

    for i in range(1,nx):
        if i % 2 == 0:
            Al[i]=4/3
        else:
            Al[i]=2/3

    Al=Al*dx

    # Controller
    if sys['ctrl_on'] == 1:
        U_sf = Al @ (sys['Kvu1'] * u_sf_a) + Al @ (sys['Kvv1'] * v_sf_a)
        U_of = Al @ (sys['Kvu1'] * u_hat_a) + Al @ (sys['Kvv1'] * v_hat_a)
    else:
        U_sf = 0
        U_of = 0

    # System and observer dynamics
    Delta = sys['Delta']
    u_sf_a[0] = sys['q'] * v_sf_a[0]
    v_sf_a[-1] = U_sf
    u_of_a[0] = sys['q'] * v_sf_a[0]
    v_of_a[-1] = U_of
    u_hat_a[0] = sys['q'] * v_sf_a[0]
    v_hat_a[-1] = U_of

    # Spatial derivatives
    u_sf_x = np.concatenate([[(u_sf_a[1] - u_sf_a[0]) / Delta],
                             (u_sf_a[:-3] - 4 * u_sf_a[1:-2] + 3 * u_sf_a[2:-1]) / (2 * Delta)])

    v_sf_x = np.concatenate([(-3 * v_sf_a[1:N_grid-2] + 4 * v_sf_a[2:N_grid-1] - v_sf_a[3:N_grid]) / (2 * Delta),
                             [(v_sf_a[-1] - v_sf_a[-2]) / Delta]])

    u_of_x = np.concatenate([[(u_of_a[1] - u_of_a[0]) / Delta],
                             (u_of_a[:-3] - 4 * u_of_a[1:-2] + 3 * u_of_a[2:-1]) / (2 * Delta)])
    v_of_x = np.concatenate([(-3 * v_of_a[1:N_grid-2] + 4 * v_of_a[2:N_grid-1] - v_of_a[3:N_grid]) / (2 * Delta),
                             [(v_of_a[-1] - v_of_a[-2]) / Delta]])

    u_hat_x = np.concatenate([[(u_hat_a[1] - u_hat_a[0]) / Delta],
                              (u_hat_a[:-3] - 4 * u_hat_a[1:-2] + 3 * u_hat_a[2:-1]) / (2 * Delta)])
    v_hat_x = np.concatenate([(-3 * v_hat_a[1:N_grid-2] + 4 * v_hat_a[2:N_grid-1] - v_hat_a[3:N_grid]) / (2 * Delta),
                             [(v_hat_a[-1] - v_hat_a[-2]) / Delta]])

    u_sf_t = -Lambda[1:N+1] * u_sf_x +DDDelta[1:N+1] * u_sf+ c_1[1:N+1] * v_sf

    v_sf_t = mu[1:N+1] * v_sf_x + c_2[1:N+1] * u_sf

    u_of_t = -Lambda[1:N+1] * u_of_x +DDDelta[1:N+1] * u_of+ c_1[1:N+1] * v_of
    v_of_t = mu[1:N+1] * v_of_x + c_2[1:N+1] * u_of

    u_hat_t = -Lambda[1:N+1] * u_hat_x +DDDelta[1:N+1] * u_hat+ c_1[1:N+1] * v_hat + sys['p1'][1:N+1] * (v_of_a[0] - v_hat_a[0])
    v_hat_t = mu[1:N+1] * v_hat_x + c_2[1:N+1] * u_hat + sys['p2'][1:N+1] * (v_of_a[0] - v_hat_a[0])

    U_sf_f_t = sys['gamma_U'] * (U_sf - U_sf_f)
    U_of_f_t = sys['gamma_U'] * (U_of - U_of_f)

    # Parse
    dt = np.concatenate([u_sf_t, v_sf_t, u_of_t, v_of_t, u_hat_t, v_hat_t, [U_sf_f_t, U_of_f_t]])
    return dt

def ode_func(t,x):
    return ode_chap_08_06(t,x,sys)

def count_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class BranchNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.conv1 = torch.nn.Conv2d(6, 16, 5, stride=2) # 4输入
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, 5, stride=2)
        self.fc1 = torch.nn.Linear(70688, 1028)
        self.fc2 = torch.nn.Linear(1028, 256)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1, self.shape, self.shape))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # print("x.shape", x.shape)
        return x


def zeroToNan(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if j <= i:
                x[i][j] = float('nan')
    return x

def zeroToNan1(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if j >= i:
                x[i][j] = float('nan')
    return x

def zeroToNan0(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if j == i:
                x[i][j] = 0
    return x

# Initialize system parameters
sys = {}
sys['ctrl_on'] = 1
sys['second_order'] = 1
sys['RGB_color1'] = [1, 0, 0]

sys['RGB_color2'] = [0, 0, 1]
sys['RGB_color3'] = [0, 1, 0]
sys['style1'] = '--'
sys['style2'] = '-.'
sys['style3'] = '-'

# Build out two examples
uarr = []
varr = []
uhatarr = []
vhatarr = []

uofarr = []
uhatofarr = []
vofarr = []
vhatofarr = []


k1arr = []
k2arr = []
k1hatarr = []
k2hatarr = []

m1arr = []
m2arr = []
m1hatarr = []
m2hatarr = []

lamarr=[2,5]

# model1=torch.load("Hyperbolic_m1_2000").cuda()
# model2=torch.load("Hyperbolic_m2_2000").cuda()
# model3=torch.load("Hyperbolic_k1_2000").cuda()
# model4=torch.load("Hyperbolic_k2_2000").cuda()

# Define functions
fun = {}
for i in range(2):
    print("i=",i)
    fun['lambda'] = lambda x: lamarr[i]* x + 1
    fun['mu'] = lambda x: np.exp(lamarr[i]* x) + 2
    fun['lambda_d'] = lambda x: 0 * x + lamarr[i]
    fun['mu_d'] = lambda x: lamarr[i]* np.exp(lamarr[i] * x) + 0
    fun['c_1'] = lambda x: lamarr[i]* (np.cosh(x) + 1)
    fun['c_2'] = lambda x: lamarr[i]* (x + 1)
    fun['delta'] = lambda x: lamarr[i]* (x + 1)
    fun['q'] = lamarr[i]/ 3

    sys['fun'] = fun

    # Simulation specific parameters
    sys['N'] = 200
    sys['N_grid'] = sys['N'] + 2
    sys['N_g'] = 200
    sys['simH'] = 6
    sys['h'] = 0.01
    sys['Tspan'] = np.arange(0, sys['simH'], sys['h'])
    sys['Delta'] = 1 / (sys['N'] + 1)
    sys['xspan'] = np.linspace(0, 1, sys['N_grid'])
    sys['xspanT'] = np.arange(sys['Delta'], 1, sys['Delta'])
    sys['intArr'] = sys['Delta'] * np.array([0.5] + [1] * sys['N'] + [0.5])

    spatial = np.linspace(0, 1, sys['N_grid'], dtype=np.float32)
    spatial1 = np.linspace(0, 1, sys['N_grid'], dtype=np.float32)
    
    x = np.linspace(0, 1, sys['N_grid'])
    temporal = np.linspace(0, 5.99, 600, dtype=np.float32)
    nt=int(round(6/0.01))

    # Expanding system parameters
    sys['lambda'] = fun['lambda'](sys['xspan'])
    sys['mu'] = fun['mu'](sys['xspan'])
    sys['c_1'] = fun['c_1'](sys['xspan'])
    sys['c_2'] = fun['c_2'](sys['xspan'])
    sys['q'] = fun['q']
    sys['Ddelta']=fun['delta'](sys['xspan'])
    sys['gamma_U'] = 1

    # kernels conculation
    Kvu, Kvv = K_solver_2x2(fun, sys['N_g'])
    sys['Kvu'] = Kvu
    sys['Kvv'] = Kvv
    k1arr.append(Kvu.T)
    k2arr.append(Kvv.T)
    sys['Kvu1'] = PchipInterpolator(np.linspace(0, 1, sys['N_g']), Kvu[:, -1])(sys['xspan'])
    sys['Kvv1'] = PchipInterpolator(np.linspace(0, 1, sys['N_g']), Kvv[:, -1])(sys['xspan'])

    Paa, Pba = P_solver_2x2(fun, sys['N_g'])
    sys['Paa'] = Paa
    sys['Pba'] = Pba
    m1=Paa
    m2=Pba
    m1arr.append(Paa)
    m2arr.append(Pba)
    Pa = np.array([m1[sys['N_g'] - i - 1, -1] for i in range(sys['N_g'])])
    Pb = np.array([m2[sys['N_g'] - i - 1, -1] for i in range(sys['N_g'])])
    sys['p1'] = sys['mu'][0] * PchipInterpolator(np.linspace(0, 1, sys['N_g']), Pa)(sys['xspan'])
    sys['p2'] = sys['mu'][0] * PchipInterpolator(np.linspace(0, 1, sys['N_g']), Pb)(sys['xspan'])

    # X = 1
    # dx = 0.005
    # nx = int(round(X/dx))
    # spatial = np.linspace(0, X, nx, dtype=np.float32)
    # grids = []
    # grids.append(spatial)
    # grids.append(spatial)
    # grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    # grid1 = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    # grid = np.concatenate([grid, grid, grid], axis=1)
    # grid = torch.from_numpy(grid).cuda()

    # x_Mu1= sys['mu']
    # x_Lambda1= sys['lambda']
    # x_C11= sys['c_1']
    # x_C21= sys['c_2']
    # x_Delta1= sys['Ddelta']
    # x_Q1= sys['q']

    # x_mu1=np.repeat(x_Mu1[1:201,np.newaxis],200,axis=1)
    # x_lambda1=np.repeat(x_Lambda1[1:201,np.newaxis],200,axis=1)
    # x_c11=np.repeat(x_C11[1:201,np.newaxis],200,axis=1)
    # x_c21=np.repeat(x_C21[1:201,np.newaxis],200,axis=1)
    # x_delta1=np.repeat(x_Delta1[1:201,np.newaxis],200,axis=1)
    # AA=np.ones((200,200))
    # x_q1=x_Q1*AA

    # x_mu1 = np.array(x_mu1, dtype=np.float32)
    # x_lambda1 = np.array(x_lambda1, dtype=np.float32)
    # x_c11= np.array(x_c11, dtype=np.float32)
    # x_c21 = np.array(x_c21, dtype=np.float32)
    # x_delta1 = np.array(x_delta1, dtype=np.float32)
    # x_q1 = np.array(x_q1, dtype=np.float32)

    # x_mu = x_mu1.reshape(1, -1)
    # x_lambda = x_lambda1.reshape(1, -1)
    # x_c1 = x_c11.reshape(1, -1)
    # x_c2 = x_c21.reshape(1, -1)
    # x_delta = x_delta1.reshape(1, -1)
    # x_q= x_q1.reshape(1, -1)

    # x = np.stack((x_mu,x_lambda,x_c1,x_c2,x_delta,x_q), axis=1)
    # xval = torch.from_numpy(x.reshape(1, 6, 40000)).cuda()

    # hat_k = model3((xval, grid))
    # hat_l = model4((xval, grid))
    # hat_k_numpy = hat_k.detach().cpu().numpy()
    # hat_l_numpy = hat_l.detach().cpu().numpy()
    # hat_k=hat_k_numpy.reshape(200,200)
    # hat_l=hat_l_numpy.reshape(200,200)
    # hat_k=zeroToNan0(hat_k)
    # hat_l=zeroToNan0(hat_l)
    # k1hatarr.append(hat_k)
    # k2hatarr.append(hat_l)
    # sys['Kvu1'] = PchipInterpolator(np.linspace(0, 1, sys['N_g']), hat_k[:, -1])(sys['xspan'])
    # sys['Kvv1'] = PchipInterpolator(np.linspace(0, 1, sys['N_g']), hat_l[:, -1])(sys['xspan'])


    # hat_m1 = model1((xval, grid))
    # hat_m2 = model2((xval, grid))
    # hat_m1_numpy = hat_m1.detach().cpu().numpy()
    # hat_m2_numpy = hat_m2.detach().cpu().numpy()
    # hat_m1=hat_m1_numpy.reshape(200,200)
    # hat_m2=hat_m2_numpy.reshape(200,200)
    # hat_m1=zeroToNan0(hat_m1)
    # hat_m2=zeroToNan0(hat_m2)
    # m1hatarr.append(hat_m1)
    # m2hatarr.append(hat_m2)
    # Pa = np.array([hat_m1[sys['N_g'] - i - 1, -1] for i in range(sys['N_g'])])
    # Pb = np.array([hat_m2[sys['N_g'] - i - 1, -1] for i in range(sys['N_g'])])
    # sys['p1'] = sys['mu'][0] * PchipInterpolator(np.linspace(0, 1, sys['N_g']), Pa)(sys['xspan'])
    # sys['p2'] = sys['mu'][0] * PchipInterpolator(np.linspace(0, 1, sys['N_g']), Pb)(sys['xspan'])


    # Initial condition
    u_sf_0 = np.ones(sys['N'])
    v_sf_0 = np.sin(sys['xspanT'])
    u_of_0 = np.ones(sys['N'])
    v_of_0 = np.sin(sys['xspanT'])
    u_hat_0 = np.zeros(sys['N'])
    v_hat_0 = np.zeros(sys['N'])
    U_sf_f_0 = 0
    U_of_f_0 = 0
    x0 = np.concatenate([u_sf_0, v_sf_0, u_of_0, v_of_0, u_hat_0, v_hat_0, [U_sf_f_0, U_of_f_0]])

    # solve PDE with Learning operator

    dim_x = 1
    m = sys['N_grid']
    branch = [m, 256, 256]
    trunk = [dim_x, 128, 256]
    # branch = [m, 128, 256, 256]
    # trunk = [dim_x, 128,128, 256]
    activation = "relu"
    kernel = "Glorot normal"
    class DeepONetModified(nn.Module):
        def __init__(self, branch, trunk, activation, kernel, projection):
            super(DeepONetModified, self).__init__()
            self.net1 = dde.nn.DeepONetCartesianProd(branch, trunk, activation, kernel).cuda()
            self.fc1 = nn.Linear(m, m)
            self.fc2 = nn.Linear(m, m)
            self.net2 = dde.nn.DeepONetCartesianProd(branch, trunk, activation, kernel).cuda()
            self.fc3 = nn.Linear(m,128)
            self.fc4 = nn.Linear(128, 64)
            self.fc5 = nn.Linear(64, projection)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x, grid = x[0], x[1]
            x = self.net1((x, grid))
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.net2((x, grid))
            x = self.fc3(x)
            x = self.fc4(x)
            x = self.fc5(x)
            return x

    projection  = 1
    model = DeepONetModified(branch, trunk, activation, kernel, projection)
    # model = torch.load("Hyperbolic_U_12302").cuda()
    model = torch.load("U_110").cuda()
    result = solve_ivp(ode_func, [sys['Tspan'][0], sys['Tspan'][-1]], x0, method='RK45', t_eval=sys['Tspan'])    #, method='RK45', t_eval=sys['Tspan']

    t_log = result.t
    x_log = result.y.T

    numT = len(t_log)
    N = sys['N']
    N_grid = sys['N_grid']

    xx = x_log[:,0:6*N].reshape(numT, 6, N)

    xx_a = np.zeros((numT, 6, N + 2))
    xx_a[:, :, 0] = 2 * xx[:, :, 0] - xx[:, :, 1]
    xx_a[:, :, -1] = 2 * xx[:, :, -1] - xx[:, :, -2]
    xx_a[:, :, 1:N+1] = xx

    u_sf_a = xx_a[:, 0, :]
    v_sf_a = xx_a[:, 1, :]
    u_of_a = xx_a[:, 2, :]
    v_of_a = xx_a[:, 3, :]
    u_hat_a = xx_a[:, 4, :]
    v_hat_a = xx_a[:, 5, :]


    # ######### slove PDE
    # result = solve_ivp(ode_func, [sys['Tspan'][0], sys['Tspan'][-1]], x0, method='RK45', t_eval=sys['Tspan'])    #, method='RK45', t_eval=sys['Tspan']
    # t_log = result.t
    # x_log = result.y.T
    # numT = len(t_log)
    # N = sys['N']
    # N_grid = sys['N_grid']
    # xx = x_log[:,0:6*N].reshape(numT, 6, N)
    # xx_a = np.zeros((numT, 6, N + 2))
    # xx_a[:, :, 0] = 2 * xx[:, :, 0] - xx[:, :, 1]
    # xx_a[:, :, -1] = 2 * xx[:, :, -1] - xx[:, :, -2]
    # xx_a[:, :, 1:N+1] = xx
    # u_sf_a = xx_a[:, 0, :]
    # v_sf_a = xx_a[:, 1, :]
    # u_of_a = xx_a[:, 2, :]
    # v_of_a = xx_a[:, 3, :]
    # u_hat_a = xx_a[:, 4, :]
    # v_hat_a = xx_a[:, 5, :]
    # np.savetxt("u_sf_a2.dat", u_sf_a)
    # np.savetxt("v_sf_a2.dat", v_sf_a)
    # np.savetxt("u_of_a2.dat", u_of_a)
    # np.savetxt("v_of_a2.dat", v_of_a)
    # np.savetxt("u_hat_a2.dat", u_hat_a)
    # np.savetxt("v_hat_a2.dat", v_hat_a)

############ KERNEL PLOT ###########
res = 10
fig = plt.figure(figsize=(10, 4))
subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig = subfigs
subfig.subplots_adjust(left=0.02, bottom=0, right=1, top=1, wspace=-0.05, hspace=0)

subfig.suptitle(r"$k_1(x, \xi)$ for $\Gamma=2,5$")
meshx, mesht = np.meshgrid(spatial, spatial)

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
for axis in [ax[0].xaxis, ax[0].yaxis, ax[0].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))
for axis in [ax[1].xaxis, ax[1].yaxis, ax[1].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))

ax[0].plot_surface(meshx, mesht, zeroToNan1(k1arr[0]), edgecolor="black",lw=0.2, rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True)
ax[0].view_init(10,15)
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$\xi$")
ax[0].set_zlabel(r"$k_1(x,\xi)$", rotation=90)
ax[0].set_xticks([0, 0.5, 1])
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


ax[1].plot_surface(meshx, mesht, zeroToNan1(k1arr[1]), edgecolor="black",lw=0.2,rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel(r"$\xi$")
ax[1].set_zlabel(r"$k_1(x,\xi)$", rotation=90)
ax[1].set_xticks([0, 0.5, 1])
ax[0].zaxis.set_rotate_label(False)
ax[1].zaxis.set_rotate_label(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.show()

res = 10
fig = plt.figure(figsize=(10, 4))
subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig = subfigs
subfig.subplots_adjust(left=0.02, bottom=0, right=1, top=1, wspace=-0.05, hspace=0)

subfig.suptitle(r"$\hat k_1(x, \xi)$ for $\Gamma=2,5$")
meshx, mesht = np.meshgrid(spatial, spatial)

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
for axis in [ax[0].xaxis, ax[0].yaxis, ax[0].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))
for axis in [ax[1].xaxis, ax[1].yaxis, ax[1].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))

ax[0].plot_surface(meshx, mesht, zeroToNan1(k1hatarr[0]), edgecolor="black",lw=0.2, rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True)
ax[0].view_init(10,15)
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$\xi$")
ax[0].set_zlabel(r"$\hat k_1(x,\xi)$", rotation=90)
ax[0].set_xticks([0, 0.5, 1])
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax[1].plot_surface(meshx, mesht, zeroToNan1(k1hatarr[1]), edgecolor="black",lw=0.2,rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel(r"$\xi$")
ax[1].set_zlabel(r"$\hat k_1(x,\xi)$", rotation=90)
ax[1].set_xticks([0, 0.5, 1])
ax[0].zaxis.set_rotate_label(False)
ax[1].zaxis.set_rotate_label(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()

############ k1- hat k1  KERNEL PLOT ###########
res = 10
fig = plt.figure(figsize=(10,4))
subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig = subfigs
subfig.subplots_adjust(left=0.02, bottom=0, right=1, top=1, wspace=-0.05, hspace=0)

subfig.suptitle(r"$k_1(x, \xi)-\hat k_1(x, \xi)$ for $\Gamma=2,5$")
meshx, mesht = np.meshgrid(spatial, spatial)

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
for axis in [ax[0].xaxis, ax[0].yaxis, ax[0].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))
for axis in [ax[1].xaxis, ax[1].yaxis, ax[1].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))

ax[0].plot_surface(meshx, mesht, zeroToNan1(k1arr[0]-k1hatarr[0]), edgecolor="black",lw=0.2, rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True)
ax[0].view_init(10,15)
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$\xi$")
ax[0].set_zlabel(r"$k_1(x,\xi)-\hat k_1(x,\xi)$", rotation=90)
ax[0].set_xticks([0, 0.5, 1])
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax[1].plot_surface(meshx, mesht, zeroToNan1(k1arr[1]-k1hatarr[1]), edgecolor="black",lw=0.2,rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel(r"$\xi$")
ax[1].set_zlabel(r"$k_1(x,\xi)-\hat k_1(x,\xi)$", rotation=90)
ax[1].set_xticks([0, 0.5, 1])
ax[0].zaxis.set_rotate_label(False)
ax[1].zaxis.set_rotate_label(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()

############ m1 KERNEL PLOT ###########
res = 10
fig = plt.figure(figsize=(10, 4))
subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig = subfigs
subfig.subplots_adjust(left=0.02, bottom=0, right=1, top=1, wspace=-0.05, hspace=0)

subfig.suptitle(r"$m_1(x, \xi)$ for $\Gamma=2,5$")
meshx, mesht = np.meshgrid(spatial, spatial)

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
for axis in [ax[0].xaxis, ax[0].yaxis, ax[0].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))
for axis in [ax[1].xaxis, ax[1].yaxis, ax[1].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))

ax[0].plot_surface(meshx, mesht, zeroToNan(m1arr[0]), edgecolor="black",lw=0.2, rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True)
ax[0].view_init(10,15)
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$\xi$")
ax[0].set_zlabel(r"$m_1(x,\xi)$", rotation=90)
ax[0].set_xticks([0, 0.5, 1])
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


ax[1].plot_surface(meshx, mesht, zeroToNan(m1arr[1]), edgecolor="black",lw=0.2,rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel(r"$\xi$")
ax[1].set_zlabel(r"$m_1(x,\xi)$", rotation=90)
ax[1].set_xticks([0, 0.5, 1])
ax[0].zaxis.set_rotate_label(False)
ax[1].zaxis.set_rotate_label(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()

res = 10
fig = plt.figure(figsize=(10, 4))
subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig = subfigs
subfig.subplots_adjust(left=0.02, bottom=0, right=1, top=1, wspace=-0.05, hspace=0)

subfig.suptitle(r"$\hat m_1(x, \xi)$ for $\Gamma=2,5$")
meshx, mesht = np.meshgrid(spatial, spatial)

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
for axis in [ax[0].xaxis, ax[0].yaxis, ax[0].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))
for axis in [ax[1].xaxis, ax[1].yaxis, ax[1].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))

ax[0].plot_surface(meshx, mesht, zeroToNan(m1hatarr[0]), edgecolor="black",lw=0.2, rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True)
ax[0].view_init(10,15)
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$\xi$")
ax[0].set_zlabel(r"$\hat m_1(x,\xi)$", rotation=90)
ax[0].set_xticks([0, 0.5, 1])
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax[1].plot_surface(meshx, mesht, zeroToNan(m1hatarr[1]), edgecolor="black",lw=0.2,rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel(r"$\xi$")
ax[1].set_zlabel(r"$\hat m_1(x,\xi)$", rotation=90)
ax[1].set_xticks([0, 0.5, 1])
ax[0].zaxis.set_rotate_label(False)
ax[1].zaxis.set_rotate_label(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()

res = 10
fig = plt.figure(figsize=(10, 4))
subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig = subfigs
subfig.subplots_adjust(left=0.02, bottom=0, right=1, top=1, wspace=-0.05, hspace=0)

subfig.suptitle(r"$m_1(x, \xi)-\hat m_1(x, \xi)$ for $\Gamma=2,5$")
meshx, mesht = np.meshgrid(spatial, spatial)

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
for axis in [ax[0].xaxis, ax[0].yaxis, ax[0].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))
for axis in [ax[1].xaxis, ax[1].yaxis, ax[1].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0))

ax[0].plot_surface(meshx, mesht, zeroToNan(m1arr[0]-m1hatarr[0]), edgecolor="black",lw=0.2, rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True)
ax[0].view_init(10,15)
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$\xi$")
ax[0].set_zlabel(r"$m_1(x,\xi)-\hat m_1(x,\xi)$", rotation=90)
ax[0].set_xticks([0, 0.5, 1])
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax[1].plot_surface(meshx, mesht, zeroToNan(m1arr[1]-m1hatarr[1]), edgecolor="black",lw=0.2,rstride=res, cstride=res,
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel(r"$\xi$")
ax[1].set_zlabel(r"$m_1(x,\xi)-\hat m_1(x,\xi)$", rotation=90)
ax[1].set_xticks([0, 0.5, 1])
ax[0].zaxis.set_rotate_label(False)
ax[1].zaxis.set_rotate_label(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()