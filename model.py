
import chemderiv
import matplotlib.pyplot as plt
import math
from chem_utils import cranknicolson
import numpy as np
import netCDF4
import time

debug_level = 3

##########################################################
##################### CLASS DEF ##########################
##########################################################

class DebugBreak(BaseException):
    pass

class OddTypeError(BaseException):
    pass

class Concentrations(object):
    def __init__(self, params):
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

    def test_type(self, error_msg):
        # type checking utility for use in debugging
        if not all(type(v) is float for v in self.values.values()):
            types = np.unique([str(type(v)) for v in self.values.values()])
            raise OddTypeError('{} : {}'.format(error_msg, types))

    def getArgList(self, i, j, k):
        # set up ordered argList for input inmto chemfunction
        # the order is determined by the ordered list given in the parameter class
        labels = self.chemicals;
        argList = []
        for label in self.params.orderedLabels:
            argList.append(self.values[label][i,j,k])
        return argList

    def emit(self):
        # Uses emissions data to alter concentrations, dealing with unit
        # conversions accordingly
        for chemical in self.chemicals:
            emis = self.params.emissions[chemical]
            scalar = (self.params.time_step * (1e5)**2 * 1/self.params.height * 6.022e23)
            d_conc = scalar * emis
            # molecules/cm^3 =  E (mol km^-2 hr^-1) * dt (hr) * 1e5^2 (km/cm)^2 * 1/height (1/cm) * molec/mole
            self.values[chemical] = self.values[chemical] + d_conc

    def deposit(self):
        # Uses deposition velocity data to alter concentrations, dealing with unit
        # conversions accordingly
        for chemical in self.chemicals:
            vdep = self.params.depVel[chemical]
            d_conc = self.values[chemical] * vdep * (self.params.time_step * 3600) * 1/self.params.height;
            # molec/cm3 = vdep {'cm/s'} * concentration (molec/cm3) * dt (sec)/height (cm)
            self.values[chemical] = self.values[chemical] + \
                ( np.zeros([self.params.xdim, self.params.ydim, self.params.zdim]) * d_conc )

    def advect(self):
        # Use crank nicolson to deal with advection and diffusion
        for chemical in self.chemicals:
            self.values[chemical] = advection_diffusion(  self.values[chemical], self.params.U, self.params.V,
                                                    self.params.W, self.params.bc[chemical], del_t = self.params.time_step)

    def spinup(self):
        # Spin up over the specified time period
        hours = self.params.hour_generator(self.params.spin_up_time_range)
        for t in np.arange(self.params.initial_time, self.params.spinup_duration + self.params.initial_time, self.params.time_step):
            hour = next(hours)
            self.emit()
            self.deposit()
            self.advect()
            self.chem(hour)

    def steady_state_calc(self, values, chemical):
        # calculates the steady state concentration of the desired chemical according to the
        # corresponding steady state function stored in a dictionary within the params instance
        # the ss_dependancies property of the params instance gives the dictionary keys of the
        # chemicals that need to be used as arguments in the steady state function
        f = self.params.steady_state_func[chemical]
        args = [values[dependancy] for dependancy in self.params.ss_dependancies[chemical]]
        return f(*args)

    def chem(self, hour):
        # updates the concentraions using the chemderiv function, iterating over each grid cell
        # the results of the grid cells are independant, this could be parallelized
        for i in range(self.params.xdim):
            for j in range(self.params.ydim):
                for k in range(self.params.zdim):
                    # combine the static arguments with the chemical arguments for the cythonized kinetics function call
                    args = [self.params.time_step, hour, self.params.temp(hour), self.params.c_m, self.params.height]
                    args += self.getArgList(i, j, k)
                    results = chemderiv.chem_solver(*args)
                    # call relevant methods to calculate the steady state concentrations of chemicals if necessary
                    for chemical in results.keys():
                        if self.params.steady_state_bool[chemical]:
                            self.values[chemical][i,j,k] = self.steady_state_calc(results, chemical)
                        else:
                            self.values[chemical][i,j,k] = results[chemical]

class ModelParams(object):

    def __init__(self, xdim=5, ydim=5, zdim=5):
    # Initiates and Organizes Constant Properties for Model
        self.time_step = 0.5
        self.initial_time = 12
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
        self.conc_H2O = 10 # filler
        self.pHOX = 10 # filler
        self.RH = 10 # filler
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.c_m = 2.5e19; # air densitys in molecules/cm3
        self.spinup_duration = 2; # hours to spin up model

        default_matrix = lambda val: val * np.ones([self.xdim, self.ydim, self.zdim])
        default_bc = { '-x': 0,'+x': 0,'-y': 0,'+y': 0,'-z': 0,'+z': 0}
        self.bc = { 'conc_NO2' : default_bc,
                    'conc_APN' : default_bc,
                    'conc_AP'  : default_bc,
                    'conc_NO'  : default_bc,
                    'conc_O3'  : default_bc,
                    'conc_HNO3': default_bc,
                    'conc_HO'  : default_bc,
                    'conc_HO2' : default_bc,
                    'conc_PROD': default_bc}
        self.emissions = {
                            'conc_NO2' : default_matrix(1),
                            'conc_APN' : default_matrix(1),
                            'conc_AP'  : default_matrix(1),
                            'conc_NO'  : default_matrix(1),
                            'conc_O3'  : default_matrix(1),
                            'conc_HNO3': default_matrix(1),
                            'conc_HO'  : default_matrix(1),
                            'conc_HO2' : default_matrix(1),
                            'conc_PROD': default_matrix(1)}
        self.initial = {
                            'conc_NO2' : default_matrix(1),
                            'conc_APN' : default_matrix(1),
                            'conc_AP'  : default_matrix(1),
                            'conc_NO'  : default_matrix(1),
                            'conc_O3'  : default_matrix(1),
                            'conc_HNO3': default_matrix(1),
                            'conc_HO'  : default_matrix(1),
                            'conc_HO2' : default_matrix(1),
                            'conc_PROD': default_matrix(1) }
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
                            'conc_HO'  : lambda no2, TEMP: self.pHOX/no2 * self.c_m * rates['TROE'](1.49e-30,1.8,2.58e-11,0, TEMP, self.c_m),
                            'conc_HO2' : lambda no2, no: self.pHOX * self.RH / no2 * no * self.c_m}
        self.ss_dependancies = {
                            'conc_HO'  : ['conc_NO2', 'TEMP'] ,
                            'conc_HO2' : ['conc_NO2','conc_NO'] }
        self.time_range = np.arange(self.initial_time, self.final_time, self.time_step)
        self.spin_up_time_range = np.arange(self.initial_time - self.spinup_duration, self.initial_time, self.time_step)
        self.U = 2
        self.V = 2
        self.W = 2

    def temp(self, t):
        # Pearson Type III Model for Diurnal Temperature Cycle
        Tmin, Trange, a = 15, 15, 9
        if t > 24:
            t = t - 24
        if t <= 5:
            t += 10
        elif t >= 14:
            t -= 14
        elif t > 5 and t < 14:
            t -= 14
        if t < 0:
            gam = 0.24
        else:
            gam = 0.8
        return 273 + Tmin + Trange * (math.exp(-gam * t) * (1 + t/a) ** (gam*a))

    def hour_generator(self, time_range):
        # used to generate the hours when calling the chem function
        for t in time_range:
            yield t

##########################################################
######################### RATES ##########################
##########################################################

def TROE( k0_300K,  n,  kinf_300K,  m,  TEMP,  C_M):
    zt_help = 300.0 / TEMP;
    k0_T    = k0_300K   * zt_help ** n * C_M   # k_0   at current T
    kinf_T  = kinf_300K * zt_help ** m          # k_inf at current T
    k_ratio = k0_T/kinf_T
    return k0_T/(1.0 + k_ratio)*0.6 ** (1.0 / (1.0+log10(k_ratio)**2))

rates = { 'TROE' : TROE }

##########################################################
######################### MAIN  ##########################
##########################################################

def main(*args):
    """ Main Driver Function """

    # initiate classes
    p = ModelParams()
    conc = Concentrations(p)

    # create netcdf4 file to write data
    nc_file = netCDF4.Dataset('SM_Data.nc','w', format='NETCDF4_CLASSIC')
    nc_file.history = 'Created '+ time.ctime(time.time())

    # create the cartesian and time dimesions, which must be Also
    # creates as variables
    xdim =      nc_file.createDimension('xdim',p.xdim)
    ydim =      nc_file.createDimension('ydim',p.ydim)
    zdim =      nc_file.createDimension('zdim',p.zdim)
    timedim =   nc_file.createDimension('time', len(p.time_range))

    # create and assign variables xdim, ydim, zdim, and time
    # which will serve as the the chemical dimensions
    # where xdim, ydim, and zdim are first converted into meters
    # before added as variables
    xvar = nc_file.createVariable('xdim', np.int32, ('xdim'))
    xvar[:] = p.height/100 * np.arange(p.xdim)
    xvar.units = 'meters'

    yvar = nc_file.createVariable('ydim', np.int32, ('ydim'))
    yvar[:] = p.height/100 * np.arange(p.ydim)
    yvar.units = 'meters'

    zvar = nc_file.createVariable('zdim', np.int32, ('zdim'))
    zvar[:] = p.height/100 * np.arange(p.zdim)
    zvar.units = 'meters'

    # create the dimimension in hours
    tiemvar = nc_file.createVariable('time', np.float32, ('time'))
    tiemvar[:] = p.time_range
    tiemvar.units = 'hours'

    # create a variable for each chemical quantity with dimensions
    # time, xdim, ydim, and zdim
    cvars = dict()
    for chemical in conc.chemicals:
        cvars[chemical] = nc_file.createVariable(chemical, np.float64, ('xdim','ydim','zdim','time'))
        cvars[chemical].units = 'molec/cm3'

    # cycle over time range, writing data at each time point to the netcdf file
    t_index = 0
    hours = p.hour_generator(p.time_range)
    for t in p.time_range:
        hour = next(hours)
        for chemical in conc.chemicals:
            cvars[chemical][:,:,:,t_index] = conc.values[chemical]
        # cyling emision/deposition/advection+diffusion/chemistry
        conc.emit()
        conc.deposit()
        conc.advect()
        conc.chem(hour)
        t_index += 1

    # writes file
    nc_file.close()

if __name__ == '__main__':
    main()
