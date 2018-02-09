
import numpy as np
import sys
sys.path.append('../')

from Case import *

GLOB_c_m = 2.5e19 # air dens in molecules/cm3
GLOB_U = 5 #m/s
GLOB_V = 0
GLOB_W = 0

GLOB_dx = 100000 # 1km, in cm
GLOB_dy = 100000
GLOB_dz = 100000
GLOB_Diff = 0 # cm^2/s
GLOB_CN_timestep = 6 # in seconds for CN (6 * gridsize in km)

GLOB_time_step = 0.5 # in hours for operator split
GLOB_chemical_dt = 1 # in seconds for within chem func
GLOB_initial_time = 12
GLOB_final_time = 14
GLOB_height = GLOB_dz # centimeter box GLOB_height
GLOB_spinup_duration = 0 # hours to spin up model

GLOB_orderedLabels = ['conc_O3', 'conc_NO2', 'conc_NO', 'conc_HNO3', 'conc_HO']
GLOB_time_range = np.arange(GLOB_initial_time, GLOB_final_time, GLOB_time_step)
GLOB_spin_up_time_range = np.arange(GLOB_initial_time - GLOB_spinup_duration, GLOB_initial_time, GLOB_time_step)

##########################################################
##################### STEADY STATE #######################
##########################################################

steady_state_bool = {'conc_O3': 0, 'conc_NO2': 0, 'conc_NO': 0, 'conc_HNO3': 0, 'conc_HO': 0}
fixed = {'conc_O3': 1, 'conc_NO2': 0, 'conc_NO': 0, 'conc_HNO3': 0, 'conc_HO': 1}
steady_state_func = {'conc_O3': None, 'conc_NO2': None, 'conc_NO': None, 'conc_HNO3': None, 'conc_HO': None}

##########################################################
##################### MAKE THE CASE ######################
##########################################################

xdim = 3;
ydim = 3;
zdim = 1;

conc_NO = 10 * 2.5e19/1e9;
conc_NO2 = 10 * 2.5e19/1e9;
conc_HO = 0.3 * 2.5e19/1e12;
conc_HNO3 = 10 * 2.5e19/1e12;
conc_O3 = 60 * 2.5e19/1e12;

init = {'conc_NO2' : conc_NO2 * np.ones([xdim, ydim, zdim]),
        'conc_NO'  : conc_NO * np.ones([xdim, ydim, zdim]),
        'conc_O3'  : conc_O3 * np.ones([xdim, ydim, zdim]),
        'conc_HNO3': conc_HNO3 * np.ones([xdim, ydim, zdim]),
        'conc_HO'  : conc_HO * np.ones([xdim, ydim, zdim])}

init['conc_NO2'][xdim//2][ydim//2][zdim//2] = 30 * 2.5e19/1e9
init['conc_NO'][xdim//2][ydim//2][zdim//2] = 30 * 2.5e19/1e9

emis = {'conc_NO2' : 0 * np.ones([xdim, ydim, zdim]),
        'conc_NO'  : 0 * np.ones([xdim, ydim, zdim]),
        'conc_O3'  : 0 * np.ones([xdim, ydim, zdim]),
        'conc_HNO3': 0 * np.ones([xdim, ydim, zdim]),
        'conc_HO'  : 0 * np.ones([xdim, ydim, zdim])}

simple_bc = {   'conc_NO2' : make_uniform_bc(conc_NO2),
                'conc_NO'  : make_uniform_bc(conc_NO),
                'conc_O3'  : make_uniform_bc(conc_O3),
                'conc_HNO3': make_uniform_bc(conc_HNO3),
                'conc_HO'  : make_uniform_bc(conc_HO)}

no_vd = {   'conc_NO2' : 0,
            'conc_NO'  : 0,
            'conc_O3'  : 0,
            'conc_HNO3': 0,
            'conc_HO'  : 0}

case = Case( bc = simple_bc, initial = init,
            emis = emis, depVel = no_vd, ss_bool = steady_state_bool, ss_func = steady_state_func, fx = fixed )

GLOB_case_dict['NOxTest'] = case;
