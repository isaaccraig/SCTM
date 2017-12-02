
import chemderiv
import matplotlib.pyplot as plt
import math
from chem_utils import advection_diffusion
import numpy as np
import netCDF4
import time
import argparse

debug_level = 3

##########################################################
##################### CLASS DEF ##########################
##########################################################

class DebugBreak(BaseException):
    pass

class OddTypeError(BaseException):
    pass

class Concentrations(object):
    def __init__(self, params, depositon_on = True, emission_on=True, advection_on=True, chemistry_on=True):
        # Constructor
        self.advection_on = advection_on
        self.chemistry_on = chemistry_on
        self.emission_on = emission_on
        self.depositon_on = depositon_on

        self.chem_applied, self.advected, self.emitted, self.deposited, self.spun_up = False, False, False, False, False

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

    """
    def emit(self):
        # Uses emissions data to alter concentrations, dealing with unit
        # conversions accordingly
        if self.emission_on:
            self.emitted = True
            for chemical in self.chemicals:
                emis = self.params.emissions[chemical]
                scalar = (self.params.time_step * (1e5)**2 * 1/self.params.height * 6.022e23)
                d_conc = scalar * emis
                # molecules/cm^3 =  E (mol km^-2 hr^-1) * dt (hr) * 1e5^2 (km/cm)^2 * 1/height (1/cm) * molec/mole
                self.values[chemical] = self.values[chemical] + d_conc"""

    def emit(self):
        # Uses emissions data to alter concentrations, dealing with unit
        # conversions accordingly
        if self.emission_on:
            self.emitted = True
            for chemical in self.chemicals:
                emis = self.params.emissions[chemical]
                scalar = (self.params.time_step * 3600)
                d_conc = scalar * emis
                # molecules/cm^3 =  E (molec cm^-3 s^-1) * dt (hr) * 3600s/hr
                self.values[chemical] = self.values[chemical] + d_conc

    def deposit(self):
        # Uses deposition velocity data to alter concentrations, dealing with unit
        # conversions accordingly
        if self.depositon_on:
            self.deposited = True
            for chemical in self.chemicals:
                vdep = self.params.depVel[chemical]
                d_conc = self.values[chemical] * vdep * (self.params.time_step * 3600) * 1/self.params.height
                # molec/cm3 = vdep {'cm/s'} * concentration (molec/cm3) * dt (sec)/height (cm)
                self.values[chemical] += d_conc

    def advect(self):
        # Use crank nicolson to deal with advection and diffusion
        if self.advection_on:
            self.advected = True
            for chemical in self.chemicals:
                self.values[chemical] = advection_diffusion(  self.values[chemical], self.params.U, self.params.V,
                                                    self.params.W, self.params.bc[chemical], del_t = self.params.time_step)

    def round_and_check_neg(self, t):
        for chemical in self.chemicals:
            for i in range(self.values[chemical].shape[0]):
                for j in range(self.values[chemical].shape[1]):
                    for k in range(self.values[chemical].shape[2]):
                        self.values[chemical][i,j,k] = round(self.values[chemical][i,j,k], 10)
                        if self.values[chemical][i,j,k] < 0:
                            print("{}-{}-{}-{}-{} = {}".format(chemical, i, j, k, t, self.values[chemical][i,j,k]))
                            raise OddTypeError

    def spinup(self):
        # Spin up over the specified time period
        self.spun_up = True
        for t in np.arange(self.params.initial_time, self.params.spinup_duration + self.params.initial_time, self.params.time_step):
            print('spinning up at time {}\n'.format(t))

            self.round_and_check_neg(t)
            print('emitting at time {}\n'.format(t))
            self.emit()

            self.round_and_check_neg(t)
            print('depositing at time {}\n'.format(t))
            self.deposit()

            self.round_and_check_neg(t)
            print('advecting at time {}\n'.format(t))
            self.advect()

            self.round_and_check_neg(t)
            print('chemistry at time {}\n'.format(t))
            self.chem(t)

    def chem(self, hour):
        # updates the concentraions using the chemderiv function, iterating over each grid cell
        # the results of the grid cells are independant, this could be parallelized
        if self.chemistry_on:
            self.chem_applied = True
            for t in np.arange(0, int(self.params.time_step * 3600), self.params.chemical_dt):
                if t%10 == 0:
                    print('running chem at time t = {} seconds'.format(t))
                for i in range(self.params.xdim):
                    for j in range(self.params.ydim):
                        for k in range(self.params.zdim):
                            # combine the static arguments with the chemical arguments for the cythonized kinetics function call
                            args = [self.params.time_step, hour, self.params.temp(hour), self.params.c_m, self.params.height]
                            args += self.getArgList(i, j, k)
                            results = chemderiv.chem_solver(*args)
                            # call relevant methods to calculate the steady state concentrations of chemicals if necessary
                            for chemical in self.chemicals:
                                Ci = self.values[chemical][i,j,k]
                                self.values[chemical][i,j,k] += results[chemical] * self.params.chemical_dt # dt in seconds
                                dCdt = results[chemical] * self.params.chemical_dt
                                if self.values[chemical][i,j,k] < 0:
                                    print("{}-{}-{}-{}-{} = {}".format(chemical, i, j, k, t, self.values[chemical][i,j,k]))
                                    print("Ci = {}, dCdt * dt = {}".format(Ci, dCdt))
                                    raise OddTypeError
                            for chemical in self.chemicals:
                                if self.params.steady_state_bool[chemical]:
                                    f = self.params.steady_state_func[chemical]
                                    args = [self.values[dependancy][i,j,k] for dependancy in self.params.ss_dependancies[chemical]]
                                    self.values[chemical][i,j,k] = f(*args)
                                    if self.values[chemical][i,j,k] < 0:
                                        print("{}-{}-{}-{}-{} = {}".format(chemical, i, j, k, t, self.values[chemical][i,j,k]))
                                        print("Ci = {}, dCdt * dt = {}".format(Ci, dCdt))
                                        raise OddTypeError


default_bc = { '-x': 0,'+x': 0,'-y': 0,'+y': 0,'-z': 0,'+z': 0}
default_matrix = lambda val : val * np.ones([5, 5, 5])

bc = {  'conc_NO2' : default_bc,
        'conc_APN' : default_bc,
        'conc_AP'  : default_bc,
        'conc_NO'  : default_bc,
        'conc_O3'  : default_bc,
        'conc_HNO3': default_bc,
        'conc_HO'  : default_bc,
        'conc_HO2' : default_bc,
        'conc_PROD': default_bc}

depVel = {
                    'conc_NO2' : 0,
                    'conc_APN' : 0,
                    'conc_AP'  : 0,
                    'conc_NO'  : 0,
                    'conc_O3'  : 0,
                    'conc_HNO3': 0,
                    'conc_HO'  : 0,
                    'conc_HO2' : 0,
                    'conc_PROD': 0}

initial = lambda xdim, ydim, zdim: { 'conc_NO2' : 0.45 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                     'conc_APN' : 300 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                     'conc_AP'  : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                     'conc_NO'  : 0.15 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                     'conc_O3'  : 60 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                     'conc_HNO3': 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                     'conc_HO'  : 0.28 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                     'conc_HO2' : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                     'conc_PROD': 0 * np.ones([xdim, ydim, zdim])}
emissions = {       # molec/cm3s
                    'conc_NO2' : default_matrix(0),
                    'conc_APN' : default_matrix(0),
                    'conc_AP'  : default_matrix(5e6), # Lafranchi 2009 obs : 1e6-10e6 molec/cm3s
                    'conc_NO'  : default_matrix(1e6),
                    'conc_O3'  : default_matrix(0),
                    'conc_HNO3': default_matrix(0),
                    'conc_HO'  : default_matrix(0),
                    'conc_HO2' : default_matrix(0),
                    'conc_PROD': default_matrix(0)}

class ModelParams(object):

    def __init__(self, xdim=5, ydim=5, zdim=5, emissions=emissions ,initial=initial(5,5,5), bc=bc, depVel=depVel, spinup_duration=0.5):
    # Initiates and Organizes Constant Properties for Model
        self.time_step = 0.5
        self.chemical_dt = 0.1 # seconds
        self.initial_time = 12
        self.final_time = 18
        self.c_m = 2.5e19; # air dens in molecules/cm3

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

        # Dillon (2002) VALUE FOR PHOX
        self.pHOX = 2.25 * self.c_m/1e9 * 1/3600  # ppb/hr * C_M/1e9 = molec/cm^3hr * hr/3600s = molec/cm3s

        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.spinup_duration = spinup_duration; # hours to spin up model

        self.bc = bc
        self.emissions = emissions # mol km^-2 hr^-1
        self.initial = initial # molec/cm3
        self.depVel = depVel # cm/s

        self.steady_state_bool = {
                            'conc_NO2' : 0,
                            'conc_APN' : 0,
                            'conc_AP'  : 0,
                            'conc_NO'  : 0,
                            'conc_O3'  : 0,
                            'conc_HNO3': 0,
                            'conc_HO'  : 1,
                            'conc_HO2' : 1,
                            'conc_PROD': 0}

        def calc_ho(no2):
            if no2 == 0:
                raise error('Divide by zero in HO ss : no2 is zero')
            return self.pHOX/(no2 * self.c_m * 1.1E-11) # Sander et al. (2003)

        self.steady_state_func = {
            'conc_HO'  : calc_ho,
            'conc_HO2' : lambda ro2: ro2}

        self.ss_dependancies = { 'conc_HO'  : ['conc_NO2'] , 'conc_HO2' : ['conc_AP'] }
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

##########################################################
######################### MAIN  ##########################
##########################################################

def get_args():
    """ Get the command line arguments and return an instance of an argparse namespace """
    parser = argparse.ArgumentParser(description='Returns a netcdf file of Chemical concentrations')
    parser.add_argument('--filename', help='filename to save to')
    return parser.parse_args()

def main():
    """ Main Driver Function """

    filename = str(get_args().filename)

    # initiate classes
    p = ModelParams()
    conc = Concentrations(p)

    # create netcdf4 file to write data
    nc_file = netCDF4.Dataset('{}.nc'.format(filename),'w', format='NETCDF4_CLASSIC')
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

    for t in p.time_range:
        print('now cycling... at t = {}'.format(t))
        for chemical in conc.chemicals:
            cvars[chemical][:,:,:,t_index] = conc.values[chemical]
        # cyling emision/deposition/advection+diffusion/chemistry

        """for chemical in conc.chemicals:
            print('\t {} : ppb \n'.format(chemical))
            print(conc.values[chemical] * 1e9/2.5e19)"""

        print('Good So Far')

        conc.round_and_check_neg(t)
        print('emitting at time {}\n'.format(t))
        conc.emit()

        print('Good So Far e')

        conc.round_and_check_neg(t)
        print('depositing at time {}\n'.format(t))
        conc.deposit()

        conc.round_and_check_neg(t)
        print('advecting at time {}\n'.format(t))
        conc.advect()

        conc.round_and_check_neg(t)
        print('chemistry at time {}\n'.format(t))
        conc.chem(t)

        t_index += 1

    # writes file
    nc_file.close()

if __name__ == '__main__':
    main()
