
import chemderiv
import matplotlib.pyplot as plt
import math
from chem_utils import cranknicolson
import numpy as np

debug_level = 3

class Concentrations(object):
    def __init__(self, params, xdim = 10, ydim = 10, zdim = 10):
        # Constructor
        self.values = params.initial
        # Initial is a dictionary whose keys are the string names of
        # chemicals and the values are xdim x ydim matrices of Concentrations
        self.chemicals = params.initial.keys()
        # The keys of this dictionary are the names of the chemicals
        self.params = params
        # the container class of all physical parameters
        self.spinup()
        # Spin up within Constructor to obtain concentrations for later use

    def getArgList(self, i, j, k):
        # set up ordered argList for input into chemfunction
        # the order is determined by the ordered list given in the parameter class
        labels = self.chemicals;
        argList = []
        for label in self.params.orderedLabels:
            argList.append(self.values[label][i,j,k])
        return argList

    def emit(self):
        # Uses emissions data to alter concentrations, dealing with unit
        # conversions accordingly
        if debug_level > 2:
            print('emitting')
        for chemical in self.chemicals:
            emis = self.params.emissions[chemical]
            scalar = (self.params.time_step * (1e5)**2 * 1/self.params.height * 6.022e23)
            d_conc = np.dot(emis, scalar)
            # molecules/cm^3 =  E (mol km^-2 hr^-1) * dt (hr) * 1e5^2 (km/cm)^2 * 1/height (1/cm) * molec/mole
            self.values[chemical] = self.values[chemical] + \
                ( np.zeros([self.params.xdim, self.params.ydim, self.params.zdim]) * d_conc )

    def deposit(self):
        # Uses deposition velocity data to alter concentrations, dealing with unit
        # conversions accordingly
        if debug_level > 2:
            print('depositing')
        for chemical in self.chemicals:
            vdep = self.params.depVel[chemical]
            d_conc = self.values[chemical] * vdep * (self.params.time_step * 3600) * 1/self.params.height;
            # molec/cm3 = vdep {'cm/s'} * concentration (molec/cm3) * dt (sec)/height (cm)
            self.values[chemical] = self.values[chemical] + \
                ( np.zeros([self.params.xdim, self.params.ydim, self.params.zdim]) * d_conc )

    def advect(self):
        # Use crank nicolson to deal with advection and diffusion
        if debug_level > 2:
            print('advecting')
        for chemical in self.chemicals:
            self.values[chemical] = cranknicolson(  self.values[chemical], self.params.U, self.params.V,
                                                    self.params.W, self.params.time_step, self.params.time_step)

    def spinup(self):
        # Spin up over the specified time period
        for t in np.arange(self.params.initial_time, self.params.spinup_duration + self.params.initial_time, self.params.time_step):
            if debug_level > 2:
                print('on spinup time {}'.format(t))
            self.emit()
            self.deposit()
            self.advect()
            self.chem()

    def steady_state_calc(self, values, chemical):
        f = self.params.steady_state_func[chemical]
        args = [values[dependancy] for dependancy in self.params.ss_dependancies[chemical]]
        return f(*args)

    def chem(self):
        hours = self.params.hour_generator()
        for i in range(self.params.xdim):
            for j in range(self.params.ydim):
                for k in range(self.params.zdim):
                    hour = next(hours)
                    args =  [self.params.time_step, hour, self.params.temp(hour), self.params.c_m, self.params.height]
                    args += self.getArgList(i, j, k)
                    results = chemderiv.chem_solver(*args)
                    for chemical in results.keys():
                        if self.params.steady_state_bool[chemical]:
                            self.values[chemical][i,j,k] = self.steady_state_calc(results, chemical)
                        else:
                            self.values[chemical][i,j,k] = results[chemical]

class ModelParams(object):
    def __init__(self):
    # Initiates Constant Properties
        self.time_step = 0.5
        self.initial_time = 12
        self.pHOX = 'filler'
        self.final_time = 18
        self.orderedLabels = ['conc_O3',
                              'conc_NO2',
                              'conc_NO',
                              'conc_AP',
                              'conc_APN',
                              'conc_HNO3',
                              'conc_HO',
                              'conc_HO2',
                              'conc_PROD']
        self.height = 100000; # centimeter height
        self.xdim = 10
        self.ydim = 10
        self.zdim = 10
        self.c_m = 2.5e19; # air densitys in molecules/cm3
        self.spinup_duration = 2; # hours to spin up model
        self.emissions = {
                            'conc_NO2' : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_APN' : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_AP'  : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_NO'  : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_O3'  : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_HNO3': np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_HO'  : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_HO2' : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_PROD' : np.zeros([self.xdim, self.ydim, self.zdim])}
        self.initial = {
                            'conc_NO2' : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_APN' : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_AP'  : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_NO'  : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_O3'  : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_HNO3': np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_HO'  : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_HO2' : np.zeros([self.xdim, self.ydim, self.zdim]),
                            'conc_PROD' : np.zeros([self.xdim, self.ydim, self.zdim])}
        self.depVel = {
                            'conc_NO2' : 0,
                            'conc_APN' : 0,
                            'conc_AP'  : 0,
                            'conc_NO'  : 0,
                            'conc_O3'  : 0,
                            'conc_HNO3': 0,
                            'conc_HO'  : 0,
                            'conc_HO2' : 0,
                            'conc_PROD': 0}
        self.steady_state_bool = {
                            'conc_NO2' : 0,
                            'conc_APN' : 0,
                            'conc_AP'  : 0,
                            'conc_NO'  : 0,
                            'conc_O3'  : 0,
                            'conc_HNO3': 0,
                            'conc_HO'  : 1,
                            'conc_HO2' : 1,
                            'conc_PROD' : 0}
        self.steady_state_func = {
                            'conc_HO'  : lambda no2: self.pHOX/ no2 * self.c_m * k,
                            'conc_HO2' : lambda no2, no: self.pHOX * self.rh/ no2 * no * self.c_m}
        self.ss_dependancies = {
                            'conc_HO'  : ['conc_NO2'] ,
                            'conc_HO2' : ['conc_NO2','conc_NO'] }
        self.time_range = np.arange(self.time_step, self.final_time, self.time_step)
        self.U = 2
        self.V = 2
        self.W = 2

    def temp(self, t):
        # Pearson Type III Model for Diurnal Temperature Cycle
        Tmin, Trange, a = 15, 15, 9
        if t > 24:
            t = t - 24

        if t < 5 or t == 5:
            t += 10
        elif t > 14 or t == 14:
            t -= 14
        elif t > 5 and t < 14:
            t -= 14

        if t < 0:
            gam = 0.24
        else:
            gam = 0.8

        return 273 + Tmin + Trange * (math.exp(-gam * t) * (1 + t/a) ** (gam*a))

    def hour_generator(self):
        for t in self.time_range:
            yield t

def main(*args):
    """ Main Driver Function """

    p = ModelParams()
    conc = Concentrations(p)

    for t in range(p.initial_time, p.final_time, p.time_step):
        conc.emit()
        conc.deposit()
        conc.advect()
        conc.chem()

if __name__ == '__main__':
    main()

    # save values to a net_cdf_file
