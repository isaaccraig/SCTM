
import numpy as np
from ../cases import *

##########################################################
################ Physical Parameters #####################
##########################################################

GLOB_c_m = 2.5e19 # air dens in molecules/cm3
GLOB_pHOX = 2.25 * GLOB_c_m/1e9 * 1/3600  # Dillon (2002) GLOB_VALGLOB_UE FOR PHOX # ppb/hr * C_M/1e9 = molec/cm^3hr * hr/3600s = molec/cm3s
GLOB_U = 10
GLOB_V = 0
GLOB_W = 0

##########################################################
################ Model Parameters ########################
##########################################################

GLOB_dx = 100000 # 1km, in cm
GLOB_dy = 100000
GLOB_dz = 100000
GLOB_Diff = 1e4 # cm^2/s
GLOB_CN_timestep = 100 # in seconds for CN

GLOB_time_step = 0.5 # in hours for operator split
GLOB_chemical_dt = 1 # in seconds for within chem func
GLOB_initial_time = 12
GLOB_final_time = 14
GLOB_height = GLOB_dz # centimeter box GLOB_height
GLOB_spinup_duration = 0 # hours to spin up model

GLOB_orderedLabels = ['conc_O3','conc_NO2','conc_NO','conc_AP','conc_APN','conc_HNO3','conc_HO', 'conc_HO2','conc_PROD']
GLOB_time_range = np.arange(GLOB_initial_time, GLOB_final_time, GLOB_time_step)
GLOB_spin_up_time_range = np.arange(GLOB_initial_time - GLOB_spinup_duration, GLOB_initial_time, GLOB_time_step)

##########################################################
##################### STEADY STATE #######################
##########################################################

def sscalc_ho(values, i, j, k):
    if values["conc_NO2"][i,j,k] == 0:
        raise error('Divide by zero in HO ss : no2 is zero')
    return GLOB_pHOX/(values["conc_NO2"][i,j,k] * GLOB_c_m * 1.1E-11) # Sander et al. (2003)

def sscalc_ho2(values, i, j, k):
    return values["conc_AP"][i,j,k]

steady_state_bool = { 'conc_NO2' : 0, 'conc_APN' : 0,'conc_AP'  : 0,'conc_NO'  : 0,\
        'conc_O3'  : 0, 'conc_HNO3': 0,'conc_HO'  : 1,'conc_HO2' : 1,'conc_PROD': 0}

fixed = { 'conc_NO2' : 0, 'conc_APN' : 0,'conc_AP'  : 0,'conc_NO'  : 0,\
        'conc_O3'  : 0, 'conc_HNO3': 0,'conc_HO'  : 1,'conc_HO2' : 0,'conc_PROD': 0}

steady_state_func = { 'conc_NO2' : None , 'conc_APN' : None,'conc_AP'  : None, \
        'conc_NO'  : None, 'conc_O3'  : None, 'conc_HNO3': None,'conc_HO'  : sscalc_ho,\
        'conc_HO2' : sscalc_ho2,'conc_PROD': None}

##############################################################
################### CASE CONSTRUCTOR UTILS ###################
##############################################################

make_realisitic_initial = lambda xdim, ydim, zdim: { 'conc_NO2' : 0.45 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_APN' : 300 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_AP'  : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_NO'  : 0.15 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_O3'  : 60 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_HNO3': 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO'  : 0.28 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO2' : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_PROD': 0 * np.ones([xdim, ydim, zdim])}

def make_realisitic_centered_NOX (xdim, ydim, zdim):
    d = {   'conc_NO2' : 10 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
            'conc_APN' : 300 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
            'conc_AP'  : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
            'conc_NO'  : 10 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
            'conc_O3'  : 60 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
            'conc_HNO3': 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
            'conc_HO'  : 0.28 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
            'conc_HO2' : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
            'conc_PROD': 0 * np.ones([xdim, ydim, zdim])
            }
    d['conc_NO2'][xdim//2][ydim//2][zdim//2] = 0.45 * 2.5e19/1e9
    d['conc_NO'][xdim//2][ydim//2][zdim//2] = 0.15 * 2.5e19/1e9
    return d

make_realisitic_emis = lambda xdim, ydim, zdim: { 'conc_NO2' : 0 * np.ones([xdim, ydim, zdim]),
                                                'conc_APN' : 0 * np.ones([xdim, ydim, zdim]),
                                                'conc_AP'  : 5e6 * np.ones([xdim, ydim, zdim]),
                                                'conc_NO'  : 1e6 * np.ones([xdim, ydim, zdim]),
                                                'conc_O3'  : 0 * np.ones([xdim, ydim, zdim]),
                                                'conc_HNO3': 0 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO'  : 0 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO2' : 0 * np.ones([xdim, ydim, zdim]),
                                                'conc_PROD': 0 * np.ones([xdim, ydim, zdim])}

##############################################################
######################### BASE CASE ##########################
##############################################################

simple_bc = {   'conc_NO2' : make_uniform_bc(0),
                'conc_APN' : make_uniform_bc(0),
                'conc_AP'  : make_uniform_bc(0),
                'conc_NO'  : make_uniform_bc(0),
                'conc_O3'  : make_uniform_bc(0),
                'conc_HNO3': make_uniform_bc(0),
                'conc_HO'  : make_uniform_bc(0),
                'conc_HO2' : make_uniform_bc(0),
                'conc_PROD': make_uniform_bc(0)}

simple_vd = {   'conc_NO2' : 0,
                'conc_APN' : 1,
                'conc_AP'  : 0,
                'conc_NO'  : 0,
                'conc_O3'  : 0,
                'conc_HNO3': 1,
                'conc_HO'  : 0,
                'conc_HO2' : 0,
                'conc_PROD': 0}

##############################################################
######################### CASE DICT ##########################
##############################################################

testing_case = Case( bc = simple_bc,
                     initial = make_realisitic_initial(15,15,1),
                     emis = make_realisitic_emis(15,15,1),
                     depVel = simple_vd )

centered_case = Case( bc = simple_bc,
                     initial = make_realisitic_centered_NOX(15,15,1),
                     emis = make_realisitic_emis(15,15,1),
                     depVel = simple_vd )

case_dict['testing'] = testing_case;
case_dict['centered'] = centered_case;
