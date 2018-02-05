
from cnic import CN
from forward_time_central_space import forward_time_central_space
from forwardeuler import forward_space_time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import argparse

method_dict = { "CN": CN,
                "FE": forward_space_time,
                "FTCS": forward_time_central_space}

class Function_Stepper:
    def __init__(self, func, init_state):
        self.state = init_state
        self.time_elapsed = 0
        self.func = func

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        self.state = self.func(self.state)


def get_full_x( x_range, Nx, Ny, Nz ):
    axis = np.zeros(Nx * Ny * Nz)
    i = 0
    for x in x_range:
        for _ in range(Ny * Nz):
            axis[i] = x
            i += 1
    return axis

def get_full_y( y_range, Nx, Ny, Nz ):
    axis = np.zeros(Nx * Ny * Nz)
    i = 0
    for _ in range(Nz):
        for y in y_range:
            for __ in range(Nx):
                axis[i] = y
                i += 1
    return axis

def get_full_z( z_range, Nx, Ny, Nz ):
    axis = np.zeros(Nx * Ny * Nz)
    i = 0
    for _ in range(Nx * Ny):
        for z in z_range:
            axis[i] = z
            i += 1
    return axis

def flatten( C, Nx, Ny, Nz ):
    flat = np.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                flat[ k + Nz*j + Nz*Ny*i ] = C[i,j,k]
    return np.array(flat)

def get_args():
    """ Get the command line arguments and return an instance of an argparse namespace """
    parser = argparse.ArgumentParser(description='Runs test animation of numerical advection diffusion')
    parser.add_argument('--method', help='method to use')
    return parser.parse_args()

def main():
    #------------------------------------------------------------
    # set up initial state
    init_state = np.ones([3,3,3])
    init_state[1,1,1] = 10

    Nx = init_state.shape[0]
    Ny = init_state.shape[1]
    Nz = init_state.shape[2]

    x_range = range(Nx)
    y_range = range(Ny)
    z_range = range(Nz)

    x_axislist = np.array(get_full_x(x_range, Nx, Ny, Nz))
    y_axislist = np.array(get_full_y(y_range, Nx, Ny, Nz))
    z_axislist = np.array(get_full_z(z_range, Nx, Ny, Nz))

    method_nm = str(get_args().method)

    try:
        function = method_dict[method_nm]
    except:
        print("ERROR: UNKNOWN FUNCTION TO TEST")
        exit(1);

    stepper = Function_Stepper(function, init_state)
    dt = 1. / 30 # 30fps

    #------------------------------------------------------------
    # set up figure and animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_axislist, y_axislist, z_axislist, c=[], cmap=plt.hot())

    def animate(i, scatter, dt):
        """perform animation step"""
        stepper.step(dt)
        flattened = flatten(stepper.state, Nx, Ny, Nz)
        scatter.set_array(flattened)
        return scatter,

    ani = animation.FuncAnimation(fig, animate, frames=600, fargs = (scatter, dt))
    plt.show()

if __name__ == '__main__':
    main()
