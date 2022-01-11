import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import glob
from numpy import genfromtxt
import math
import numpy as np
from scipy import linalg
from scipy.linalg import logm, expm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import pandas as pd

def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=1):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.
    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    #print(eigvals)

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z



def scale_volume(A, scaling_factor):
    # A= (nthroot(scale,3)^2).*A;
    return ((scaling_factor**(1/float(3)))**2)*A


def scale_axes(A, scaling_factors):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    s=s*scaling_factors
    #s=s*1/np.sqrt(scaling_factors)
    return np.matmul(np.matmul(u, np.diag(s)), vh)


def get_volume(A):
    # scale=prod(sqrt(eig(M)))*(4.0/3.0)*pi;
    w,v = np.linalg.eig(A)

    return (math.sqrt(w[0])*math.sqrt(w[1])*math.sqrt(w[2]))*(4.0/3.0)*math.pi



def logmap(X,S):
    v,d = np.linalg.eig(np.linalg.inv(S)*X)
    U = S* v* np.diag(np.log(np.diag(d)))*(np.linalg.inv(v))
    return U

def expmap(U, S):
    v,d = np.linalg.eig(np.linalg.inv(S)*U)
    X = S* v* np.diag(np.exp(np.diag(d)))*(np.linalg.inv(v))
    return X


def get_logeuclidean_distance(A,B):
    # d = norm(logm(A) - logm(B), 'fro');
    return np.linalg.norm(logm(A)-logm(B), 'fro')