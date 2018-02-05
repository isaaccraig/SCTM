
import numpy as np

#############################################################
######################## CASE OBJECT ########################
#############################################################

GLOB_case_dict = {};

make_uniform_bc = lambda val : { '-x': val,'+x': val,'-y': val,'+y': val,'-z': val, '+z': val}

class Case(object):
    def __init__(self, bc, initial, emis, depVel, ss_bool, ss_func, fx):
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

        self.steady_state_bool = ss_bool
        self.fixed = fx
        self.steady_state_func = ss_func
