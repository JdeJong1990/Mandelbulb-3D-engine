# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 19:15:12 2023

@author: Joost

Class that creates an image of the Mandelbulb fractal, as seen from 
a desired position. 
"""
import os
import time

import matplotlib.pyplot as plt
import matplotlib.image as pli
import numpy as np
from numpy import pi as pi
from numpy import linalg as LA

from Jtoolbox import rotx, rotz, mand2


class Mandelbulb:
    """A Mandelbulb object creates an image of the famous Mandelbulb fractal.
    Set the following parameters to deviate from the default image
    res     :Resolution of the output image, res x res pixels 
    nmarch  :Number of steps that every probing ray travels to find the object
    order   :Number of iterations to approximate fractal, higher order means more detail
    power   :Power of the polynomial that describes the fractal. Invluences symmetries.
    ele     :Elevational angle under which the fractal is seen
    azi     :Azimuthal angle to spin the fractal
    """
    # always executed when the class is being initiated. 
    def __init__(self, res=200, nmarch=200, order=6, power=8, proximity=3.0, ele=0, azi=0):               #here are the inputs for creating the object
        self.res = res
        self.nmarch = nmarch
        self.order = order
        self.power = power
        self.proximity = proximity
        self.ele = ele
        self.azi = azi
        self.campos = np.array([0,proximity,0])

    
    def __str__(self):                      #define a string that is returned when you try to print the object
        return(
            "Mandelbulb object\n"
               +f"Resolution: {self.res}x{self.res}"
               )    

    def visualize(self):
        """visualize is a method that determines the projection of the mandelbulb.
        It determines the depthmap without making a render.
        Set all parameters of the object before running this method. 
        """
        #compile camera: prepare vectors for the origin and direction of probing rays
        vx = np.linspace(-0.5,0.5,self.res)
        vy = np.linspace(-0.5,0.5,self.res)

        xx, yy = np.meshgrid(vx,vy, indexing="ij")
        zz = np.ones(self.res**2)

        xx = np.reshape(xx,[self.res**2,1])
        yy = np.reshape(yy,[self.res**2,1])
        zz = np.reshape(zz,[self.res**2,1])
        
        dirs = np.vstack((xx.T,yy.T,zz.T)).T
        
        #Rotate ray directions (dirs) and camera position (campos) to desired orientation and position
        dirs = np.divide(dirs,np.reshape(LA.norm(dirs.T,axis=0),[self.res**2,1]))
        dirs = np.matmul(dirs,rotx(pi/2))

        dirs = np.matmul(dirs,rotx(self.ele))
        campos = np.matmul(self.campos,rotx(self.ele))

        dirs = np.matmul(dirs,  rotz(self.azi))
        campos = np.matmul(campos,rotz(self.azi))

        #Define positions on the probing rays
        self.pos = np.vstack((campos[0]*np.ones(self.res**2),
                       campos[1]*np.ones(self.res**2),
                       campos[2]*np.ones(self.res**2))).T

        start = time.time()
        #Run over the rays through "Ray marching". step size is determined by mand2(pos)
        for ii in range(self.nmarch):
            #Determine step for the current positions using the distance estimator mand2(pos)
            step = (mand2(self.pos,self.order,self.power))*0.4
            
            #Multipy step with 0 if pos gets to far from the origin
            step *= (np.reshape(LA.norm(self.pos.T,axis=0),[self.pos.shape[0],1]) < 4)      
            
            #Update positions on the ray
            self.pos += dirs*abs(step)
            print(f"Remaining iterations: {self.nmarch-ii}")
            
        end = time.time()
        self.elapsed_time=round(end-start,ndigits=None)
        print(f"Computation time: {self.elapsed_time} s")

        self.fname = str(time.strftime("%Y%m%d%H%M%S"))+"Mandelbulb.png"
    
    def render(self):
        """This method turns a distance map into a image with lighting imitation.
        Run the visualize method first.
        """
        #Create distance map        
        self.dis = np.sqrt(np.matmul(np.square(self.pos-self.campos),np.array([1,1,1])))
        
        im = np.reshape(self.dis,[self.res,self.res])
        
        #Take a image-derivative to fake a normal vector dependend intensity
        #(Replace with 2D convolution filters later)
        ims = im[:,0:-2]-im[:,1:-1]
        ims = np.arctan(ims*250-0.1)*(im[:,0:-2]<4)*self.res/2000
        ims = np.clip(ims,0,1)
        
        #Determine figure title
        filename = os.path.basename(__file__)
        im_title = (
                  f"Marchingiterations: {self.nmarch}\n"+
                  f"Manndelbulb_iterations: {self.order}\n"+
                  f"file: {filename}\n"+
                  f"Elapsed time: {self.elapsed_time} s"
                  )
        
        #Plot figure
        fig2 = plt.figure(2)
        fig2.clf()
        ax2 = fig2.subplots(1,1)
        ax2.imshow(ims.T, cmap="gray")
        ax2.set_title(im_title)
        fig2.tight_layout()
        fig2.canvas.draw()
        plt.show()
        
        #Save figure
        plt.savefig("figure_"+self.fname)

        #Save image
        pli.imsave(self.fname, ims.T,cmap="gray")
        print("Image files saved")