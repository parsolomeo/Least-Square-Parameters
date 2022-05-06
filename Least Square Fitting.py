# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 00:21:56 2021

@author: srpdo
"""

import numpy as np
import matplotlib.pyplot as plt

def lsqf(x,y):
    
    #function calculates least square fitting parameters
    

    var_i = [ ]         #variance values
    m = y.mean()        #data set mean
    n = len(y)          #number of data
    
    for i in y:
                
        var_i.append((i-m)**2/n)  #calculates varriance
        
    var_i = np.array(var_i)
    
    a_n = np.sum(y/var_i)*np.sum(x**2/var_i) - np.sum(x*y/var_i)*np.sum(x/var_i)
    a_d = np.sum(1/var_i)*np.sum(x**2/var_i) - np.sum(x/var_i)**2
    alpha = a_n/a_d
    
    b_n = np.sum(1/var_i)*np.sum(x*y/var_i) - np.sum(x/var_i)*np.sum(y/var_i)
    b_d = np.sum(1/var_i)*np.sum(x**2/var_i) - np.sum(x/var_i)**2
    beta = b_n/b_d
    
    return [alpha, beta, var_i]

def T(x,y):
    beta = lsqf(x,y)[1]
    return -1/beta

def uncrt(x,y):
    
    beta = lsqf(x,y)[1]
    var = lsqf(x,y)[2]
    
    var_beta = np.sum(1/var) /  (np.sum(1/var)*np.sum(x**2/var) - np.sum(x/var)**2) 
    var_T = np.sqrt(var_beta)*(1/beta**2)
    
    return var_T


#data
N = [106, 80, 98, 75, 74, 73, 49, 38, 37, 22]
t = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
n = len(N)

t=np.array(t)
N = np.array(N)
lnN = np.log(N)


plt.plot(t,lnN,"r.")

alpha = lsqf(t,lnN)[0]              #intercept using the least square
beta = lsqf(t,lnN)[1]               #slope using the least square 

fit = alpha+beta*t                  #least square fit
plt.plot(t,fit)                     
T = T(t,lnN)                        
print(T)
print(uncrt(t,lnN))

A = np.exp(alpha) 
decay_dist = A * np.exp(-t/T)

