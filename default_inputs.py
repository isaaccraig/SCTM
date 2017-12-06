

import numpy as np


uniform_bc = lambda val: { '-x': val,'+x': val,'-y': val,'+y': val,'-z': val,'+z': val}
realistic_bc = {                                'conc_NO2' : uniform_bc(0.45 * 2.5e19/1e9),
                                                'conc_APN' : uniform_bc(300 * 2.5e19/1e12),
                                                'conc_AP'  : uniform_bc(23 * 2.5e19/1e12),
                                                'conc_NO'  : uniform_bc(0.15 * 2.5e19/1e9),
                                                'conc_O3'  : uniform_bc(60 * 2.5e19/1e9),
                                                'conc_HNO3': uniform_bc(23 * 2.5e19/1e12),
                                                'conc_HO'  : uniform_bc(0.28 * 2.5e19/1e12),
                                                'conc_HO2' : uniform_bc(23 * 2.5e19/1e12),
                                                'conc_PROD': uniform_bc(0)}

uniform_depVel = lambda val: { 'conc_NO2' : val, 'conc_APN' : val, 'conc_AP'  : val, 'conc_NO'  : val, 'conc_O3'  : val, 'conc_HNO3': val, 'conc_HO'  : val, 'conc_HO2' : val, 'conc_PROD': val}
no_depVel = uniform_depVel(0)
realistic_depVel = lambda val: { 'conc_NO2' : 0, 'conc_APN' : 1, 'conc_AP'  : 0, 'conc_NO'  : 0, 'conc_O3'  : 0, 'conc_HNO3': 1, 'conc_HO'  : 0, 'conc_HO2' : 0, 'conc_PROD': 0}

default_matrix = lambda val : val * np.ones([5, 5, 5])
realistic_initial = lambda xdim, ydim, zdim: {  'conc_NO2' : 0.45 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_APN' : 300 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_AP'  : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_NO'  : 0.15 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_O3'  : 60 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_HNO3': 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO'  : 0.28 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO2' : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_PROD': 0 * np.ones([xdim, ydim, zdim])}
realistic_emissions = { # molec/cm3s
                                                'conc_NO2' : default_matrix(0),
                                                'conc_APN' : default_matrix(0),
                                                'conc_AP'  : default_matrix(5e6), # Lafranchi 2009 obs : 1e6-10e6 molec/cm3s
                                                'conc_NO'  : default_matrix(1e6),
                                                'conc_O3'  : default_matrix(0),
                                                'conc_HNO3': default_matrix(0),
                                                'conc_HO'  : default_matrix(0),
                                                'conc_HO2' : default_matrix(0),
                                                'conc_PROD': default_matrix(0)}
