
import unittest
import numpy as np
import math
from chem_utils import *


del_t = 1000
D = 1e5

initial = np.zeros([10,10,10])
zerobc ={ '-x': 0,'+x': 0,'-y': 0,'+y': 0,'-z': 0,'+z': 0}
initial[5,5,5] = 100 #M
final = advection_diffusion(initial, 0, 0, 0, zerobc, del_t=del_t, del_x = 1, del_y = 1, del_z = 1, D=D)
# analytical solution = 2 * M * (4 pi D t)^(-3/2) * exp ( - (x^2 + y^2 + z^2)/4Dt )
analytical = lambda x,y,z: 2 * 100 * (4*math.pi*D*del_t) ** (-3/2) * math.exp(- ((x-5)^2 + (y-5)^2 + (z-5)^2)/(4*D*del_t))

print('x\ty\t\z\tnumerical\texpected')

for x in range(0,10):
    for y in range(0,10):
        for z in range(1,10):
            print("{}\t{}\t{}\t{}\t{}\n".format(x,y,z,final[x,y,z],analytical(x,y,z)))

class Tests(unittest.TestCase):

    def test_diffusion(self):
        initial = np.zeros([10,10,10])
        zerobc ={ '-x': 0,'+x': 0,'-y': 0,'+y': 0,'-z': 0,'+z': 0}
        initial[5,5,5] = 100 #M
        final = advection_diffusion(initial, 0, 0, 0, zerobc, del_t=del_t, del_x = 1, del_y = 1, del_z = 1, D=D)
        # analytical solution = 2 * M * (4 pi D t)^(-3/2) * exp ( - (x^2 + y^2 + z^2)/4Dt )
        analytical = lambda x,y,z: 2 * 100 * (4*math.pi*D*del_t) ** (-3/2) * math.exp(- ((x-5)^2 + (y-5)^2 + (z-5)^2)/(4*D*del_t))

        for x in range(0,10):
            for y in range(0,10):
                for z in range(1,10):
                    print('analytical solution for C({},{},{},{}) = {}', x,y,z,del_t,analytical(x,y,z))
                    print('numerical solution for C({},{},{},{}) = {}', x,y,z,del_t,final[x,y,z])
