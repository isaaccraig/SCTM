
import numpy as np
import math
from chem_utils import *

################################################################################################
############# Three Dimensional Point Source Gaussian Test for Advection Diffusion #############
################################################################################################

##### Instantaneous point source of strength S at the origin
##### Fluid velocities u, v, z in the x, y, z directions
##### Dx, Dy, Dz as turbulent diffusion constants

##### dC/dt = Dx (d**2C/dx**2) + Dy (d**2C/dy**2) + Dz (d**2C/dz**2) - u dC/dx - v dC/dy - w dC/dz
##### C(x,y,z,t=0) = S * delta(x) * delta(y) * delta(z)
##### C(x,y,z,t) = 0 as x,y,z -> inf

##### molec/cm3*sec = molec/cm3*cm*cm * (cm*cm/sec) - molec/cm3*cm * (cm/s)

################################################################################################
####################################### Parameters #############################################
################################################################################################

u = 0 #cm/s
v = 0 #cm/s
w = 0 #cm/s

S = 1e15 #molec/cm^3

D = 1e4 #cm*cm/s
del_t = 1 #s
BC = { '-x': 0,'+x': 0,'-y': 0,'+y': 0,'-z': 0, '+z': 0 }

Dx = D
Dy = D
Dz = D

final_t = 50 #s
NUM_Iter = int(final_t//del_t)

NX = 3
NY = 3
NZ = 3

del_x = 1000 #box size = 1 box is 1000cm = 10m
del_y = 1000
del_z = 1000

Volume =  del_x * del_y * del_z #molecules

################################################################################################
################################# Analytical Solution ##########################################
################################################################################################

analytical_coef = lambda t : (S * Volume) / ( 8 * ( (math.pi*t*D)**(3/2))) # molec/cm3 * s^-3/2 * (cm^2/s)^-3/2 = molec

xterm = lambda x,t: ( (x - u*t) ** 2 )/ (4*Dx*t) # cm*cm / cm*cm = unitless
yterm = lambda y,t: ( (y - v*t) ** 2 )/ (4*Dy*t)
zterm = lambda z,t: ( (z - w*t) ** 2 )/ (4*Dz*t)

analytical = lambda x, y, z, t: analytical_coef(t) * math.exp( - xterm(x,t) - yterm(y,t) - zterm(z,t) ) # in molecules

#analytical_coef = lambda x, y, z : S / ( 4 * math.pi * ( ( Dy*Dx*(z**2) + Dy*Dz*(x**2) * Dz*Dx*(y**2) )**(1/2)) )
#analytical = lambda x, y, z, t: analytical_coef(x, y, z) * math.exp( -u/(2*Dx) * ( ((x**2)/Dx + (y**2)/Dy + (z**2)/Dz)**(0.5) - x ) )

################################################################################################
######################### Set Up Grid of Results at given time t ###############################
################################################################################################

expected = np.zeros([NX, NY, NZ])
resultant = np.zeros([NX, NY, NZ])

for i in range(NX):
  for j in range(NY):
    for k in range(NZ):

        x = del_x * (i - NX//2) # index to cartesian coordinate transformations
        y = del_y * (j - NY//2)
        z = del_z * (k - NZ//2)

        expected[i,j,k] = analytical(x, y, z, final_t)

        if (i == NX//2) and (j == NY//2) and (k == NZ//2) :
          resultant[i,j,k] = S           # Dirac Delta Intial (Point Source)

        else:
          resultant[i,j,k] = 0

print("-------------- INITIAL ---------------")
print(resultant)
print(resultant.sum())

################################################################################################
################################## Run cranknicolson ###########################################
################################################################################################

t = 0
for _ in range(NUM_Iter):
  t += del_t
  #Continous Source : resultant[NX//2,NY//2,NZ//2] = S
  if (t%1 == 0):
      print("------------ RUNNING CN at time {} ------------".format(t))
  resultant = advection_diffusion(C = resultant, u = u, v = v, w = w,\
        BC = BC, del_t = del_t, del_x = del_x, del_y = del_y, del_z = del_z, D = D)

################################################################################################
################################# Check if two agreee ##########################################
################################################################################################

print("------------ RUNNING TESTS ------------")
print("-------------- EXPECTED ---------------")
print(expected)
print(expected.sum())
print("-------------- RESULTANT --------------")
print(resultant)
print(resultant.sum())
