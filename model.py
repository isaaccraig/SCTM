
import chemderiv
import matplotlib.pyplot as plt
import math
from chem_utils import advection_diffusion
from cases import *
import numpy as np
import netCDF4
import time
import argparse

##########################################################
######################## PARAMS ##########################
##########################################################

################ Physical Parameters ################
GLOB_c_m = 2.5e19 # air dens in molecules/cm3
GLOB_pHOX = 2.25 * GLOB_c_m/1e9 * 1/3600  # Dillon (2002) GLOB_VALGLOB_UE FOR PHOX # ppb/hr * C_M/1e9 = molec/cm^3hr * hr/3600s = molec/cm3s
GLOB_U = 2
GLOB_V = 2
GLOB_W = 0

################ Model Parameters ################
GLOB_time_step = 0.5 # in hours for operator split
GLOB_chemical_dt = 1 # in seconds for within chem func
GLOB_initial_time = 12
GLOB_final_time = 14
GLOB_height = 100000 # centimeter box GLOB_height
GLOB_spinup_duration = 0 # hours to spin up model

GLOB_orderedLabels = ['grid_O3','grid_NO2','grid_NO','grid_AP','grid_APN','grid_HNO3','grid_HO', 'grid_HO2','grid_PROD']
GLOB_time_range = np.arange(self.GLOB_initial_time, self.GLOB_final_time, self.GLOB_time_step)
GLOB_spin_up_time_range = np.arange(self.GLOB_initial_time - self.GLOB_spinup_duration, self.GLOB_initial_time, self.GLOB_time_step)

##########################################################
##################### CLASS DEF ##########################
##########################################################

class DebugBreak(BaseException):
    pass

class OddTypeError(BaseException):
    pass

class Grid(object):
    def __init__(self, case, xdim, ydim, zdim):
        # Constructor
        assert xdim == case.xdim
        assert ydim == case.ydim
        assert zdim == case.zdim

        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim

        self.chem_applied = False
        self.advected = False
        self.emitted = False
        self.deposited = False
        self.spun_up = False

        self.values = case.initial
        # Initial is a dictionary whose keys are the string names of
        # chemicals and the values are xdim x ydim matrices of Concentrations
        self.chemicals = case.chemicals
        # The keys of this dictionary are the names of the chemicals
        self.spinup()
        # Spin up within Constructor to obtain gridentrations for later use

    def test_type(self, error_msg):
        # type checking utility for use in debugging
        if not all(type(v) is float for v in self.values.values()):
            types = np.unique([str(type(v)) for v in self.values.values()])
            raise OddTypeError('{} : {}'.format(error_msg, types))

    def print_out(self):
        for chemical in self.chemicals:
            print('\t {} : ppb \n'.format(chemical))
            print(self.values[chemical] * 1e9/2.5e19)

    def getArgList(self, i, j, k, hour):
        # set up ordered argList for input inmto chemfunction
        # the order is determined by the ordered list given in the parameter class
        labels = self.chemicals;
        argList = []
        for label in GLOB_orderedLabels:
            argList.append(self.values[label][i,j,k])
        return [GLOB_time_step, hour, temp(hour), GLOB_c_m, GLOB_height] + argList

##########################################################
##################### STEADY STATE #######################
##########################################################

def sscalc_ho(values, i, j, k):
    if values["grid_NO2"][i,j,k] == 0:
        raise error('Divide by zero in HO ss : no2 is zero')
    return GLOB_pHOX/(values["grid_NO2"][i,j,k] * GLOB_c_m * 1.1E-11) # Sander et al. (2003)

def sscalc_ho2(values, i, j, k):
    return values["grid_RO2"][i,j,k]

steady_state_bool = { 'grid_NO2' : 0, 'grid_APN' : 0,'grid_AP'  : 0,'grid_NO'  : 0,'grid_O3'  : 0, 'grid_HNO3': 0,'grid_HO'  : 1,'grid_HO2' : 1,'grid_PROD': 0}
steady_state_func = { 'grid_NO2' : None , 'grid_APN' : None,'grid_AP'  : None,'grid_NO'  : None,'grid_O3'  : None, 'grid_HNO3': None,'grid_HO'  : sscalc_ho,'grid_HO2' : sscalc_ho2,'grid_PROD': None}

##########################################################
#################### MAIN GLOB_UTILITIES #################
##########################################################

def get_temp(t):
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

def emit(grid,case):
    # GLOB_Uses emissions data to alter gridentrations, dealing with unit
    # conversions accordingly
    grid.emitted = True
    for chemical in grid.chemicals:
        emis = case.emissions[chemical]
        scalar = (GLOB_time_step * 3600)
        d_grid = scalar * emis
        # molecules/cm^3 =  E (molec cm^-3 s^-1) * dt (hr) * 3600s/hr
        grid.values[chemical] = grid.values[chemical] + d_grid

def deposit(grid,case):
    # GLOB_Uses deposition velocity data to alter gridentrations, dealing with unit
    # conversions accordingly
    grid.deposited = True
    for chemical in grid.chemicals:
        vdep = case.depGLOB_Vel[chemical]
        d_grid = grid.values[chemical] * vdep * (GLOB_time_step * 3600) * 1/GLOB_height
        # molec/cm3 = vdep {'cm/s'} * gridentration (molec/cm3) * dt (sec)/GLOB_height (cm)
        grid.values[chemical] += d_grid

def advect(grid,case):
    # GLOB_Use crank nicolson to deal with advection and diffusion
    grid.advected = True
    for chemical in grid.chemicals:
        grid.values[chemical] = advection_diffusion(  grid.values[chemical], GLOB_U, GLOB_V,
                            GLOB_W, case.bc[chemical], del_t = GLOB_time_step)

def spinup(grid, case):
    # Spin up over the specified time period
    grid.spun_up = True
    for t in np.arange(GLOB_initial_time, GLOB_spinup_duration + GLOB_initial_time, GLOB_time_step):

        print('spinup : emitting at time {}\n'.format(t))
        emit(grid)

        print('spinup : depositing at time {}\n'.format(t))
        deposit(grid)

        print('spinup : advecting at time {}\n'.format(t))
        advect(grid)

        print('spinup: chemistry at time {}\n'.format(t))
        ssc_chem(grid, t)
        grid.test_settings()

def ssc_chem(grid,hour):
    # step size controled chem
    delt = GLOB_chemical_dt
    exit_time = chem(grid, hour, delt, 0)
    while exit_time < int(GLOB_time_step * 3600):
        delt /= 2;
        print('\nREDGLOB_UCING STEP SIZE TO {}'.format(delt))
        exit_time = chem(grid, hour, delt, exit_time)

def chem(grid, hour, delt, starting_from):
    """ updates the gridentraions using the chemderiv function, iterating over each grid cell
     the results of the grid cells are independant, this could be parallelized """
    grid.chem_applied = True

    for t in np.arange(starting_from, int(GLOB_time_step * 3600), delt):
        # print status
        if t % 50 == 0:
            print('running chem at {} seconds'.format(t))
        # iterate over all grid points
        for i in grid.xdim:
            for j in grid.ydim:
                for k in grid.zdim:
                    # combine the static arguments with the chemical arguments for the cythonized kinetics function call
                    args = grid.getArgList(i, j, k, t)
                    results = chemderiv.chem_solver(*args)

                    # determine the change in each chemical
                    for chemical in grid.chemicals if not steady_state_bool[chemical]:
                        Ci = grid.values[chemical][i,j,k]
                        dCdt = results[chemical] * delt

                        if Ci + dCdt < 0:
                            print("GLOB_WARNING NEGATIGLOB_VE: Ci = {}, dCdt * dt = {} at {}({},{},{},{})".format(Ci, dCdt, chemical, i, j, k, t))
                            return t # return iteration that failed for wrapper
                        else: # not negative
                            grid.values[chemical][i,j,k] += dCdt # update gridentrations of non-ss chemicals

                    # call relevant methods to calculate the steady state gridentrations of chemicals if necessary
                    # must be done after all non-ss are updated accordingly
                    for chemical in grid.chemicals if steady_state_bool[chemical]:
                        f = steady_state_func[chemical]
                        ss_val = f(grid.values, i, j, k)
                        if ss_val < 0:
                            print("GLOB_WARNING SS NEGATIGLOB_VE: Ci = {}, dCdt * dt = {} at {}({},{},{},{})".format(Ci, dCdt, chemical, i, j, k, t))
                            return t # return iteration that failed for wrapper
                        else:
                            grid.values[chemical][i,j,k] = ss_val # update gridentrations of ss chemicals
                    return t # return iteration for wrapper

##########################################################
######################### MAIN  ##########################
##########################################################

def get_args():
    """ Get the command line arguments and return an instance of an argparse namespace """
    parser = argparse.ArgumentParser(description='Returns a netcdf file of Chemical gridentrations')
    parser.add_argument('--case', help='case to use')
    return parser.parse_args()

def get_case():
    """ Get the case from command line arguments (calls get_args), see imported dependancy cases.py """
    case_nm = str(get_args().case)
    if case_nm not in case_dict.keys:
        raise BaseException("No known case : {}".format(case_nm))
    return case_dict[case_nm]

def main():
    """ Main Driver Function """

    case = get_case();
    grid = Grid(case,5,5,1)

    # create netcdf4 file to write data
    nc_file = netCDF4.Dataset('{}.nc'.format(filename),'w', format='NETCDF4_CLASSIC')
    nc_file.history = 'Created '+ time.ctime(time.time())

    # create the cartesian and time dimesions, which must be Also
    # creates as variables
    xdim =      nc_file.createDimension('xdim', grid.xdim)
    ydim =      nc_file.createDimension('ydim', grid.ydim)
    zdim =      nc_file.createDimension('zdim', grid.zdim)
    timedim =   nc_file.createDimension('time', len(GLOB_time_range))

    # create and assign variables xdim, ydim, zdim, and time
    # which will serve as the the chemical dimensions
    # where xdim, ydim, and zdim are first converted into meters
    # before added as variables
    xvar = nc_file.createGLOB_Variable('xdim', np.int32, ('xdim'))
    xvar[:] = GLOB_height/100 * np.arange(grid.xdim)
    xvar.units = 'meters'

    yvar = nc_file.createGLOB_Variable('ydim', np.int32, ('ydim'))
    yvar[:] = GLOB_height/100 * np.arange(grid.ydim)
    yvar.units = 'meters'

    zvar = nc_file.createGLOB_Variable('zdim', np.int32, ('zdim'))
    zvar[:] = GLOB_height/100 * np.arange(grid.zdim)
    zvar.units = 'meters'

    # create the dimimension in hours
    tiemvar = nc_file.createGLOB_Variable('time', np.float32, ('time'))
    tiemvar[:] = GLOB_time_range
    tiemvar.units = 'hours'

    # create a variable for each chemical quantity with dimensions
    # time, xdim, ydim, and zdim
    cvars = dict()
    for chemical in grid.chemicals:
        cvars[chemical] = nc_file.createGLOB_Variable(chemical, np.float64, ('xdim','ydim','zdim','time'))
        cvars[chemical].units = 'molec/cm3'

    # cycle over time range, writing data at each time point to the netcdf file
    t_index = 0

    for t in GLOB_time_range:
        print('cycling... at t = {}'.format(t))
        # write data
        for chemical in grid.chemicals:
            cvars[chemical][:,:,:,t_index] = grid.values[chemical]

        # cyling emision/deposition/advection+diffusion/chemistry
        print('emitting at time {}\n'.format(t))
        emit(grid,case)
        print('depositing at time {}\n'.format(t))
        deposit(grid,case)
        print('advecting at time {}\n'.format(t))
        advect(grid,case)
        print('chemistry at time {}\n'.format(t))
        ssc_chem(grid,t)

        grid.test_settings()
        t_index += 1

    # writes file
    nc_file.close()

if __name__ == '__main__':
    main()
