
from cnic import CN
from forward_time_central_space import forward_time_central_space
from forwardeuler import forward_space_time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

class Function_Stepper:
    def __init__(self, func, init_state):
        self.state = self.init_state
        self.time_elapsed = 0
        self.func = func

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        self.state = self.func(self.state)

def main():
    #------------------------------------------------------------
    # set up initial state
    init_state = np.ones([10,10,10])
    init_state[1,1,1] = 10

    x = range(init_state.shape[0])
    y = range(init_state.shape[1])
    z = range(init_state.shape[2])

    stepper = Function_Stepper(CN, init_state)
    dt = 1. / 30 # 30fps

    #------------------------------------------------------------
    # set up figure and animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter, = ax.scatter(x, y, z, [], cmap=plt.hot())

    def init():
        """initialize animation"""
        global stepper
        scatter.set_data(stepper.state)
        return scatter

    def animate(i):
        """perform animation step"""
        global stepper, dt
        stepper.step(dt)
        scatter.set_data(stepper.state)
        return scatter

    ani = animation.FuncAnimation(fig, animate, frames=600,
                                  interval=10, blit=True, init_func=init)

    plt.show()

if __name__ == '__main__':
    main()
