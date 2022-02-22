"""
Taken from rpa paper
@author: coelhorp
"""

from scipy.linalg import eigh
import autograd.numpy as np
from pymanopt.manifolds import Rotations
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm

from functools import partial
from scipy.optimize import minimize

ALPHA=400


def cost_function_pair_euc(M, Mtilde, Q):
    t1 = M
    t2 = np.dot(Q, np.dot(Mtilde, Q.T))
    return np.linalg.norm(t1 - t2)**2

def cost_function_pair_rie(M, Mtilde, Q):
    t1 = M
    t2 = np.dot(Q, np.dot(Mtilde, Q.T))
    return distance_riemann(t1, t2)**2

def cost_function_pair_rie_reg(M, Mtilde, Q):
    alpha=ALPHA
    #print("Using regularization with parameter alpha = %.3f" %(alpha))
    t1 = M
    t2 = np.dot(Q, np.dot(Mtilde, Q.T))
    return distance_riemann(t1, t2)**2 + alpha*np.linalg.norm(Q-np.eye(3))**2

def cost_function_pair_logeuc(M, Mtilde, Q):
    t1 = M
    t2 = np.dot(Q, np.dot(Mtilde, Q.T))
    return distance_logeuc(t1, t2)**2

def cost_function_full(Q, M, Mtilde, weights=None, dist=None):
    if weights is None:
        weights = np.ones(len(M)) 
    else:
        weights = np.array(weights)
        
    if dist is None:
        dist = 'euc'
        
    cost_function_pair = {}
    cost_function_pair['euc'] = cost_function_pair_euc
    cost_function_pair['rie'] = cost_function_pair_rie
    cost_function_pair['rie_reg'] = cost_function_pair_rie_reg 
    cost_function_pair['logeuc'] = cost_function_pair_logeuc    
        
    c = []
    for Mi, Mitilde in zip(M, Mtilde):
        ci = cost_function_pair[dist](Mi, Mitilde, Q)
        c.append(ci)
    c = np.array(c)
    
    return np.dot(c, weights)


def egrad_function_pair_rie(M, Mtilde, Q):
    Mtilde_invsqrt = invsqrtm(Mtilde)
    M_sqrt = sqrtm(M)
    term_aux = np.dot(Q, np.dot(M, Q.T))
    term_aux = np.dot(Mtilde_invsqrt, np.dot(term_aux, Mtilde_invsqrt))
    return 4 * np.dot(np.dot(Mtilde_invsqrt, logm(term_aux)), np.dot(M_sqrt, Q))


def egrad_function_pair_rie_reg(M, Mtilde, Q, alpha):
    Mtilde_invsqrt = invsqrtm(Mtilde)
    M_sqrt = sqrtm(M)
    term_aux = np.dot(Q, np.dot(M, Q.T))
    term_aux = np.dot(Mtilde_invsqrt, np.dot(term_aux, Mtilde_invsqrt))
    return 4 * np.dot(np.dot(Mtilde_invsqrt, logm(term_aux)), np.dot(M_sqrt, Q)) + 2*alpha*(Q-np.eye(3))


def egrad_function_full_rie_reg(Q, M, Mtilde, weights=None):

    if weights is None:
        weights = np.ones(len(M)) 
    else:
        weights = np.array(weights)

    g = []
    alpha=ALPHA

    for Mi, Mitilde, wi in zip(M, Mtilde, weights):
        gi = egrad_function_pair_rie_reg(Mi, Mitilde, Q, alpha)
        g.append(gi * wi)
    g = np.sum(g, axis=0)        
    
    return g


def egrad_function_full_rie(Q, M, Mtilde, weights=None):

    if weights is None:
        weights = np.ones(len(M)) 
    else:
        weights = np.array(weights)

    g = []
    for Mi, Mitilde, wi in zip(M, Mtilde, weights):
        gi = egrad_function_pair_rie(Mi, Mitilde, Q)
        g.append(gi * wi)
    g = np.sum(g, axis=0)        
    
    return g


def get_rotation_matrix(M, Mtilde, weights=None, dist=None, x=None):
    
    if dist is None:
        dist = 'rie'
    if dist is 'rie_reg':
        print("USING REGULARISATION WITH ALPHA=%.3f"%(ALPHA))
    
    n = M[0].shape[0]
        
    # (1) Instantiate a manifold
    manifold = Rotations(n)
    
    # (2) Define cost function and a problem
    if dist == 'euc':
        cost = partial(cost_function_full, M=M, Mtilde=Mtilde, weights=weights, dist=dist)    
        problem = Problem(manifold=manifold, cost=cost, verbosity=0)
    elif dist == 'rie':
        cost = partial(cost_function_full, M=M, Mtilde=Mtilde, weights=weights, dist=dist)    
        egrad = partial(egrad_function_full_rie, M=M, Mtilde=Mtilde, weights=weights) 
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)
    elif dist == 'rie_reg':
        cost = partial(cost_function_full, M=M, Mtilde=Mtilde, weights=weights, dist=dist)    
        egrad = partial(egrad_function_full_rie_reg, M=M, Mtilde=Mtilde, weights=weights) 
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)
    elif dist == 'logeuc':
        cost = partial(cost_function_full, M=M, Mtilde=Mtilde, weights=weights, dist=dist)    
        egrad = partial(egrad_function_full_logeuc, M=M, Mtilde=Mtilde, weights=weights) 
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)
        
    # (3) Instantiate a Pymanopt solver
    #solver = SteepestDescent(mingradnorm=1e-3)  
    solver = SteepestDescent(logverbosity=0, mingradnorm=1e-3)   
    
    # let Pymanopt do the rest
    #print("Using x for init in rotation matrix finding: ", x)
    Q_opt = solver.solve(problem, x=x)   
    #print("det of q_opt= %.3f" %(np.determinant(Q_opt)))
    #print(Q_opt[1])
    #Q_opt = Q_opt[0]
    
    return Q_opt



# def get_affine_matrix(M, Mtilde, weights=None, dist=None, x=None):
    
#     if dist is None:
#         dist = 'rie'
    
#     n = M[0].shape[0]
        

#     elif dist == 'rie':
#         cost = partial(cost_function_full, M=M, Mtilde=Mtilde, weights=weights, dist=dist)    
#         egrad = partial(egrad_function_full_rie, M=M, Mtilde=Mtilde, weights=weights) 
#         problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)

#         cons = ({'type': 'eq', 'fun': lambda x:  x[0] - 2 * x[1] + 2})
#         res = minimize(cost, x, jac=egrad, tol=1e-3, maxiter=100, constraints=cons)

    
#     return res
    














