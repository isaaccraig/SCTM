
import numpy as np

#############################################################
######################## CASE OBJECT ########################
#############################################################

class Case(object):
    def __init__(self, bc, initial, emis, depVel):
        self.bc = bc
        self.initial = initial #dict of matrix
        self.emissions = emis
        self.depVel = depVel
        self.chemicals = initial.keys()
        first_key = list(self.chemicals)[0]

        assert initial.keys() == emis.keys() == depVel.keys()
        assert initial[first_key].shape[0] == emis[first_key].shape[0]
        assert initial[first_key].shape[1] == emis[first_key].shape[1]
        # zdim must not be equal

        self.xdim = initial[first_key].shape[0]
        self.ydim = initial[first_key].shape[1]
        self.zdim = initial[first_key].shape[2]

##############################################################
################### CASE CONSTRUCTOR UTILS ###################
##############################################################

make_uniform_bc = lambda val : { '-x': val,'+x': val,'-y': val,'+y': val,'-z': val, '+z': val}

make_realisitic_initial = lambda xdim, ydim, zdim: { 'conc_NO2' : 0.45 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_APN' : 300 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_AP'  : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_NO'  : 0.15 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_O3'  : 60 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_HNO3': 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO'  : 0.28 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO2' : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_PROD': 0 * np.ones([xdim, ydim, zdim])}

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
                     initial = make_realisitic_initial(5,5,1),
                     emis = make_realisitic_emis(5,5,1),
                     depVel = simple_vd )

case_dict = { 'testing' : testing_case }
