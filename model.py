
import chemderiv
import matplotlib.pyplot as plt
import math
from chem_utils import advection_diffusion
import default_inputs as inp
import numpy as np
import netCDF4
import time
import argparse

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
        if not advection_on:
            print('\nSettings Reminder: No Advection')

        self.chemistry_on = chemistry_on
        if not chemistry_on:
            print('\nSettings Reminder: No Chemistry')

        self.emission_on = emission_on
        if not emission_on:
            print('\nSettings Reminder: No Emission')

        self.depositon_on = depositon_on
        if not depositon_on:
            print('\nSettings Reminder: No Deposition')

        self.chem_applied, self.advected, self.emitted, self.deposited, self.spun_up = False, False, False, False, False
        self.exit_time = 0

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

    def print_out(self):
        for chemical in self.chemicals:
            print('\t {} : ppb \n'.format(chemical))
            print(self.values[chemical] * 1e9/2.5e19)

    def test_settings(self):
        # Advection
        if not self.advection_on:
            assert(not self.advected)
        else:
            assert(self.advected)
        # Emission
        if not self.emission_on:
            assert(not self.emitted)
        else:
            assert(self.emitted)
        # Chemistry
        if not self.chemistry_on:
            assert(not self.chem_applied)
        else:
            assert(self.chem_applied)
        # Deposition
        if not self.depositon_on:
            assert(not self.deposited)
        else:
            assert(self.deposited)

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

            self.round_and_check_neg(t)
            print('spinup : emitting at time {}\n'.format(t))
            self.emit()

            self.round_and_check_neg(t)
            print('spinup : depositing at time {}\n'.format(t))
            self.deposit()

            self.round_and_check_neg(t)
            print('spinup : advecting at time {}\n'.format(t))
            self.advect()

            self.round_and_check_neg(t)
            print('spinup: chemistry at time {}\n'.format(t))
            self.ssc_chem(t)

            self.test_settings()

    def ssc_chem(self,hour):
        # step size controled chem
        delt = self.params.chemical_dt
        success = self.chem(hour, delt, 0)
        while not success:
            delt /= 2;
            print('\nREDUCING STEP SIZE TO {}'.format(delt))
            success = self.chem(hour, delt, self.exit_time)

    def chem(self, hour, delt, starting_from):
        # updates the concentraions using the chemderiv function, iterating over each grid cell
        # the results of the grid cells are independant, this could be parallelized
        if self.chemistry_on:
            self.chem_applied = True
            for t in np.arange(starting_from, int(self.params.time_step * 3600), delt):
                if t%50 == 0:
                    print('running chem at {} seconds'.format(t))
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
                                dCdt = results[chemical] * delt
                                if Ci + dCdt < 0:
                                    print("WARNING NEGATIVE: Ci = {}, dCdt * dt = {} at {}({},{},{},{})".format(Ci, dCdt, chemical, i, j, k, t))
                                    self.exit_time = t
                                    return False
                                else:
                                    self.values[chemical][i,j,k] += dCdt # dt in seconds
                            for chemical in self.chemicals:
                                if self.params.steady_state_bool[chemical]:
                                    f = self.params.steady_state_func[chemical]
                                    args = [self.values[dependancy][i,j,k] for dependancy in self.params.ss_dependancies[chemical]]
                                    ss_val = f(*args)
                                    if ss_val < 0:
                                        print("WARNING SS NEGATIVE: Ci = {}, dCdt * dt = {} at {}({},{},{},{})".format(Ci, dCdt, chemical, i, j, k, t))
                                        self.exit_time = t
                                        return False
                                    else:
                                        self.values[chemical][i,j,k] = ss_val
        return True

class ModelParams(object):

    def __init__(self, xdim=5, ydim=5, zdim=5, emissions=inp.realistic_emissions , initial=inp.realistic_initial(5,5,5), bc=inp.realistic_bc, depVel=inp.no_depVel, spinup_duration=0):
    # Initiates and Organizes Constant Properties for Model

        if spinup_duration == 0:
            print('\nSettings Reminder: No Spinup')

        self.time_step = 0.5 # in hours for operator split
        self.chemical_dt = 1 # in seconds for within chem func
        self.initial_time = 12
        self.final_time = 14
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
        self.height = 100000; # centimeter box height
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
        self.U = 2
        self.V = 2
        self.W = 0

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
        print('cycling... at t = {}'.format(t))
        # write data
        for chemical in conc.chemicals:
            cvars[chemical][:,:,:,t_index] = conc.values[chemical]

        # cyling emision/deposition/advection+diffusion/chemistry
        conc.round_and_check_neg(t)
        print('emitting at time {}\n'.format(t))
        conc.emit()

        conc.round_and_check_neg(t)
        print('depositing at time {}\n'.format(t))
        conc.deposit()

        conc.round_and_check_neg(t)
        print('advecting at time {}\n'.format(t))
        conc.advect()

        conc.round_and_check_neg(t)
        print('chemistry at time {}\n'.format(t))
        conc.ssc_chem(t)

        conc.test_settings()
        t_index += 1

    # writes file
    nc_file.close()

if __name__ == '__main__':
    main()
