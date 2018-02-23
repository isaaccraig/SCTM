
import matplotlib.pyplot as plt
import math
from ChemUtils import advection_diffusion
from Case import *
import numpy as np
import netCDF4
import time
import pdb
import argparse

##########################################################
##########################################################
##########################################################

from params.NOxTest import *
from kinetics.NOx.chemderiv import chem_solver

params_file_name = 'params/NOxTest.py' # for use in recording history
chemderiv_file_name = 'NOx/chemderiv.py' # for use in recording history
GLOB_case_nm = "NOxTest";

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
        spinup(self, case)
        # Spin up within Constructor to obtain gridentrations for later use

        self.steady_state_bool = case.steady_state_bool
        self.steady_state_func = case.steady_state_func
        self.fixed = case.fixed

    def test_type(self, error_msg):
        # type checking utility for use in debugging
        if not all(type(v) is float for v in self.values.values()):
            types = np.unique([str(type(v)) for v in self.values.values()])
            raise OddTypeError('{} : {}'.format(error_msg, types))

    def print_out(self):
        for chemical in self.chemicals:
            print('\t {} : ppb \n'.format(chemical))
            print(self.values[chemical] * 1e9/2.5e19)

    def check_neg(self, msg):
        for label in self.chemicals:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    for k in range(self.zdim):
                        if self.values[label][i,j,k] < 0:
                            if round(self.values[label][i,j,k] * 1e9/2.5e19, 2) < 0 and GLOB_NegError:
                                print("ERROR : NEGATIVE {} (= {} ppb) from {}".format(label, self.values[label][i,j,k] * 1e9/2.5e19 , msg))
                                exit(-1);
                            #self.values[label][i,j,k] = 0

    def getArgList(self, i, j, k, hour):
        # set up ordered argList for input inmto chemfunction
        # the order is determined by the ordered list given in the parameter class
        labels = self.chemicals;
        argList = []
        for label in GLOB_orderedLabels:
            argList.append(self.values[label][i,j,k])
        return [GLOB_time_step, hour, get_temp(hour), GLOB_c_m, GLOB_height] + argList

##########################################################
#################### MAIN GLOB_UTILITIES #################
##########################################################

def get_temp(t):
        # Pearson Type III Model for Diurnal Temperature Cycle
        if GLOB_consttemp:
            return GLOB_Temp

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
        vdep = case.depVel[chemical]
        d_grid = grid.values[chemical] * vdep * (GLOB_time_step * 3600) * 1/GLOB_height
        # molec/cm3 = vdep {'cm/s'} * gridentration (molec/cm3) * dt (sec)/GLOB_height (cm)
        grid.values[chemical] += d_grid

def advect(grid, case, stop):
    # GLOB_Use crank nicolson to deal with advection and diffusion
    grid.advected = True
    for chemical in grid.chemicals:
        for _ in range (int((3600*GLOB_time_step)//GLOB_CN_timestep)) :
            grid.values[chemical] = advection_diffusion( grid.values[chemical], GLOB_U, GLOB_V, \
                                GLOB_W, case.bc[chemical], del_t = GLOB_CN_timestep, del_x = GLOB_dx, \
                                del_y = GLOB_dy, del_z = GLOB_dz, D = GLOB_Diff, stop=False)

            if (grid.values[chemical] < 0).any() and GLOB_NegError:
                print("ERROR : Unstable Advection Diffusion, smaller time step needed")
                pdb.set_trace()
                exit(-1)

def spinup(grid,case):
    # Spin up over the specified time period
    grid.spun_up = True
    for t in np.arange(GLOB_initial_time, GLOB_spinup_duration + GLOB_initial_time, GLOB_time_step):

        print("---------------------------------------------------")
        print('spinup : emitting at time {}\n'.format(t))
        emit(grid)

        print("---------------------------------------------------")
        print('spinup : depositing at time {}\n'.format(t))
        deposit(grid)

        print("---------------------------------------------------")
        print('spinup : advecting at time {}\n'.format(t))
        advect(grid)

        print("---------------------------------------------------")
        print('spinup: chemistry at time {}\n'.format(t))
        ssc_chem(grid, t)

def ssc_chem(grid,hour):
    # step size controled chem
    delt = GLOB_chemical_dt
    exit_time = chem(grid, hour, delt, 0)
    while exit_time != 10000:
        delt /= 2;
        if delt < 1e-4:
            raise Exception("Requires Step Size below 1e-4")
        print('\nREDUCING STEP SIZE TO {}'.format(delt))
        exit_time = chem(grid, hour, delt, exit_time)

def chem(grid, hour, delt, starting_from):
    """ updates the gridentraions using the chemderiv function, iterating over each grid cell
     the results of the grid cells are independant, this could be parallelized """
    grid.chem_applied = True
    for t in np.arange(starting_from, int(GLOB_time_step * 3600), delt):
        # print status
        # if t % 100 == 0:
        #    print('running chem at {} seconds'.format(t))
        # iterate over all grid points
        for i in range(grid.xdim):
            for j in range(grid.ydim):
                for k in range(grid.zdim):
                    # combine the static arguments with the chemical arguments for the cythonized kinetics function call
                    #if i == xdim//2 and j == ydim//2:
                    #r    pdb.set_trace()

                    argList = grid.getArgList(i, j, k, t)
                    results = chem_solver(*argList) # from chemderiv

                    if np.round((results['conc_NO'] + results['conc_NO2'] + results['conc_HNO3']), 5) != 0:
                        print("ERROR : Lost Nitrogen!")
                        pdb.set_trace()
                        exit(-1)

                    # determine the change in each chemical
                    for chemical in [chem for chem in grid.chemicals if not grid.steady_state_bool[chem]]:
                        Ci = grid.values[chemical][i,j,k]
                        dCdt = results[chemical] * delt
                        if grid.fixed[chemical]:
                            pass
                        elif Ci + dCdt < 0 and GLOB_NegError:
                            print("WARNING NEGATIVE: Ci = {}, dCdt * dt = {} at {}({},{},{},{})".format(Ci, dCdt, chemical, i, j, k, t))
                            return t # return iteration that failed for wrapper
                        else: # not negative
                            grid.values[chemical][i,j,k] += dCdt # update gridentrations of non-ss chemicals

                    # call relevant methods to calculate the steady state gridentrations of chemicals if necessary
                    # must be done after all non-ss are updated accordingly
                    for chemical in [chem for chem in grid.chemicals if grid.steady_state_bool[chem]]:
                        f = grid.steady_state_func[chemical]
                        ss_val = f(grid.values, i, j, k)
                        if grid.fixed[chemical]:
                            pass
                        elif ss_val < 0 and GLOB_NegError:
                            print("WARNING SS NEGATIVE: Ci = {}, dCdt * dt = {} at {}({},{},{},{})".format(Ci, dCdt, chemical, i, j, k, t))
                            return t # return iteration that failed for wrapper
                        else:
                            grid.values[chemical][i,j,k] = ss_val # update gridentrations of ss chemicals
    return 10000 # return iteration for wrapper

##########################################################
######################### MAIN  ##########################
##########################################################

def get_args():
    """ Get the command line arguments and return an instance of an argparse namespace """
    parser = argparse.ArgumentParser(description='Returns a netcdf file of Chemical gridentrations')
    parser.add_argument('--case', help='case to use')
    return parser.parse_args()

def get_case(case_nm):
    if case_nm not in list(GLOB_case_dict.keys()):
        raise BaseException("No known case : {}".format(case_nm))
    return GLOB_case_dict[case_nm]

def get_filename(use_default):
    if use_default:
        return GLOB_case_nm
    else:
        return str(get_args().case)

def main():
    """ Main Driver Function """

    filename = get_filename(True)
    case = get_case(filename); # use default not command args
    grid = Grid(case, case.xdim, case.ydim, case.zdim)

    # create netcdf4 file to write data
    nc_file = netCDF4.Dataset('results/{}.nc'.format(filename),'w', format='NETCDF4_CLASSIC')
    nc_file.history =   'Created '+ time.ctime(time.time()) + \
                        'with parameters ' + params_file_name + ' and chemderiv file ' + chemderiv_file_name;

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

    # create the dimimension in hours
    tiemvar = nc_file.createVariable('time', np.float32, ('time'))
    tiemvar[:] = GLOB_time_range
    tiemvar.units = 'hours'

    zvar = nc_file.createVariable('zdim', np.int32, ('zdim'))
    zvar[:] = GLOB_height/100 * np.arange(grid.zdim)
    zvar.units = 'meters'

    yvar = nc_file.createVariable('ydim', np.int32, ('ydim'))
    yvar[:] = GLOB_height/100 * np.arange(grid.ydim)
    yvar.units = 'meters'

    xvar = nc_file.createVariable('xdim', np.int32, ('xdim'))
    xvar[:] = GLOB_height/100 * np.arange(grid.xdim)
    xvar.units = 'meters'

    # create a variable for each chemical quantity with dimensions
    # time, xdim, ydim, and zdim
    cvars = dict()
    for chemical in grid.chemicals:
        cvars[chemical] = nc_file.createVariable(chemical, np.float64, ('xdim','ydim','zdim','time'))
        cvars[chemical].units = 'molec/cm3'

    # cycle over time range, writing data at each time point to the netcdf file
    t_index = 0

    for t in GLOB_time_range:

        initialN = sum(sum(sum(grid.values['conc_NO'] + grid.values['conc_NO2'] + grid.values['conc_HNO3'])));

        if t > 12:
            stop = True
        else:
            stop = False

        print("---------------------------------------------------")
        print('cycling... at t = {}\n'.format(t))

        # write data
        for chemical in grid.chemicals:
            cvars[chemical][:,:,:,t_index] = grid.values[chemical]

        # cyling emision/deposition/advection+diffusion/chemistry
        print("---------------------------------------------------")
        print('emitting at time {}\n'.format(t))
        #emit(grid,case)
        grid.check_neg("emission")

        print("---------------------------------------------------")
        print('depositing at time {}\n'.format(t))
        #deposit(grid,case)
        grid.check_neg("deposition")

        print("---------------------------------------------------")
        print('advecting at time {}\n'.format(t))
        advect(grid,case,stop)
        grid.check_neg("advection")

        print("---------------------------------------------------")
        print('chemistry at time {}\n'.format(t))
        ssc_chem(grid,t)
        grid.check_neg("chemistry")

        print(sum(sum(sum(grid.values['conc_NO'] + grid.values['conc_NO2'] + grid.values['conc_HNO3']))));

        if np.round(initialN,0) != np.round(sum(sum(sum(grid.values['conc_NO'] + grid.values['conc_NO2'] + grid.values['conc_HNO3']))),0):
            pdb.set_trace();
            print('Lost N!')

        t_index += 1

    # writes file
    nc_file.close()

if __name__ == '__main__':
    main()
