# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:07:32 2023

@author: Joost

An example script to show how to use the Mandelbulb class
This script is dependant on:
"mandelclass_v0x.py"
"Jtoolbox.py"

Tip: Make an image with res=2000. It is worth the wait.

"""
from mandelclass_v04 import Mandelbulb
from numpy import pi as pi

#%%
#Create a Mandelbulb projection object
my_fractal = Mandelbulb()

#Set parameters
my_fractal.res = 200               #Resolution, resulting image wil be res x res pixels.
my_fractal.ele = pi/180*310        #Elevational angle to look at the Mandelbulb.

#Create the projection.
my_fractal.visualize()

#Turn the projection into an image, and save it.
my_fractal.render()


