# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:50:51 2022

@author: Joost

A couple of functions to use when making a 3D engine to visualize a sphere or 
mandelbulb fractal.
"""

import numpy as np
from numpy import linalg as LA

def rotx(angle):
    """Tranformation matrix for rotation around the x-axis

    Parameters
    ----------
    angle : float

    Returns
    -------
    3x3 NumPy array
        Rotation matrix for rotation around the x-axis with the angle.

    """
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[1,0,0],[0,cos,sin],[0,-sin,cos]])

def roty(angle):
    """Tranformation matrix for rotation around the y-axis

    Parameters
    ----------
    angle : float

    Returns
    -------
    3x3 NumPy array
        Rotation matrix for rotation around the y-axis with the angle.

    """
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos,0,-sin],[0,1,0],[sin,0,cos]])

def rotz(angle):
    """Tranformation matrix for rotation around the z-axis

    Parameters
    ----------
    angle : float

    Returns
    -------
    3x3 NumPy array
        Rotation matrix for rotation around the z-axis with the angle.

    """
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos,sin,0],[-sin,cos,0],[0,0,1]])

def Sphere(pos):
    """Distance estimator for a sphere with radius 0.5 at the origin

    Parameters
    ----------
    pos : Nx3 NumPy array
        List of position vectors.

    Returns
    -------
    Nx1 NumPy array
        a vector of distance estimations between the positions and the nearest 
        surface of the sphere.

    """
    return(np.sqrt(np.matmul(pos*pos,np.array([1,1,1]))) - 0.5)

def mand2(pos,iterations=3,power=8):
    """Distance estimator for the mandelbulb fractal

    Parameters
    ----------
    pos : Nx3 NumPy array
        List of position vectors.
    iterations : integer, optional
        Number of iterations that is done to approach the distance estimation
        to the surface of the fractal. More iterations give more detail.
        The default is 3.
    power : integer, optional
        The Power of the fractal, invluences symmetry. The default is 8.

    Returns
    -------
    Nx1 NumPy array of distance estimations between the positions and the
    nearest surface of the Mandelbulb fractal.

    """
    bailout = 1.5
    z = pos
    dr = np.ones([pos.shape[0],1])
    r = np.zeros([pos.shape[0],1])
    theta = np.zeros([pos.shape[0],1])
    phi = np.zeros([pos.shape[0],1])
    zr = np.zeros([pos.shape[0],1])
    for jj in range(iterations):
        r = np.reshape(LA.norm(z.T,axis=0),[pos.shape[0],1])
        zx = np.reshape(z[:,0],[pos.shape[0],1])
        zy = np.reshape(z[:,1],[pos.shape[0],1])
        zz = np.reshape(z[:,2],[pos.shape[0],1])

        #Convert to polar coordinates.
        theta = np.arccos(np.divide(zz,r))
        phi = np.arctan(np.divide(zy,zx))             #account for dividing by zeros?
        
        drnew = r**(power-1)*power*dr+1
        drnew = np.reshape(drnew,[pos.shape[0],1])
        dr = ((drnew)*(r<bailout)+dr*(r>=bailout))
        
        #Scale and rotate the point.
        zr = np.power(r,power)
        theta = theta*power
        phi = phi*power
        
        #Convert back to cartesian coordinates.
        znew = zr*np.vstack(((np.sin(theta)*np.cos(phi)).T,
                        (np.sin(phi)*np.sin(theta)).T,
                        np.cos(theta).T)).T
        z = ((znew+pos)*(r<bailout)+z*(r>=bailout))
    
    return(abs(0.5*np.log(r)*np.divide(r,dr)))