
import unittest
from model import *
import numpy as np
import matplotlib.pyplot as plt

base = lambda xdim, ydim, zdim: {               'conc_NO2' : 0.45 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_APN' : 300 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_AP'  : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_NO'  : 0.15 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_O3'  : 60 * 2.5e19/1e9 * np.ones([xdim, ydim, zdim]),
                                                'conc_HNO3': 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO'  : 0.28 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_HO2' : 23 * 2.5e19/1e12 * np.ones([xdim, ydim, zdim]),
                                                'conc_PROD': 0 * np.ones([xdim, ydim, zdim])}

get_base = lambda val, xdim, ydim, zdim: {      'conc_NO2' : val * np.ones([xdim, ydim, zdim]),
                                                'conc_APN' : val * np.ones([xdim, ydim, zdim]),
                                                'conc_AP'  : val * np.ones([xdim, ydim, zdim]),
                                                'conc_NO'  : val * np.ones([xdim, ydim, zdim]),
                                                'conc_O3'  : val * np.ones([xdim, ydim, zdim]),
                                                'conc_HNO3': val * np.ones([xdim, ydim, zdim]),
                                                'conc_HO'  : val * np.ones([xdim, ydim, zdim]),
                                                'conc_HO2' : val * np.ones([xdim, ydim, zdim]),
                                                'conc_PROD': val * np.ones([xdim, ydim, zdim])}

get_vd = lambda val: { 'conc_NO2' : val,
          'conc_APN' : val,
          'conc_AP'  : val,
          'conc_NO'  : val,
          'conc_O3'  : val,
          'conc_HNO3': val,
          'conc_HO'  : val,
          'conc_HO2' : val,
          'conc_PROD': val}

class Tests(unittest.TestCase):

    def test_ohss(self):
        params = ModelParams()
        oh_ss = params.steady_state_func['conc_HO']
        oh_vals = np.zeros(100)
        nox_vals = np.arange(0.1,10.1,0.1) * params.c_m / 1e9 # 0 to 5 ppb
        for n in range(100):
            nox = nox_vals[n]
            oh_vals[n] = oh_ss(nox)

        plt.plot(oh_vals, nox_vals)
        plt.title('OH steady state')
        plt.xlabel('[NO2] in molec/cm^3')
        plt.ylabel('[OH] in molec/cm^3')
        plt.show()

    def test_advection(self):
        starting = get_base(0, 5, 5, 5)
        starting_copy = get_base(0, 5, 5, 5)
        params = ModelParams(initial = starting,  spinup_duration=0.5)
        conc = Concentrations(params, depositon_on = False, emission_on=False, chemistry_on=False)

        self.assertTrue(conc.advected)
        self.assertFalse(conc.chem_applied)
        self.assertFalse(conc.emitted)
        self.assertFalse(conc.deposited)
        self.assertTrue(conc.spun_up)

        for i in range(starting_copy['conc_NO2'].shape[0]):
            for j in range(starting_copy['conc_NO2'].shape[1]):
                for k in range(starting_copy['conc_NO2'].shape[2]):
                    self.assertEqual(conc.values['conc_NO2'][i,j,k], starting_copy['conc_NO2'][i,j,k])

    def test_deposition(self):
        # intiate relevant classes
        starting = get_base(20, 5, 5, 5)
        vd = get_vd(10)
        params = ModelParams(initial = starting, depVel = vd,  spinup_duration=0.5)

        self.assertEqual(params.initial['conc_NO2'][0,0,0], starting['conc_NO2'][0,0,0])
        self.assertEqual(params.depVel['conc_NO2'], vd['conc_NO2'])

        conc = Concentrations(params, emission_on=False, advection_on=False, chemistry_on=False)
        expected = 20 + (20 * 10 * (params.time_step * 3600) * 1/params.height)

        self.assertFalse(conc.advected)
        self.assertFalse(conc.chem_applied)
        self.assertFalse(conc.emitted)
        self.assertTrue(conc.deposited)
        self.assertTrue(conc.spun_up)

        # expect one round of dep from spin up in the constructor class
        for i in range(conc.values['conc_NO2'].shape[0]):
            for j in range(conc.values['conc_NO2'].shape[1]):
                for k in range(conc.values['conc_NO2'].shape[2]):
                    self.assertEqual(conc.values['conc_NO2'][i,j,k], expected)

        # expect two rounds of dep
        conc.deposit()
        expected += expected * 10 * (params.time_step * 3600) * 1/params.height

        for i in range(conc.values['conc_NO2'].shape[0]):
            for j in range(conc.values['conc_NO2'].shape[1]):
                for k in range(conc.values['conc_NO2'].shape[2]):
                    self.assertEqual(conc.values['conc_NO2'][i,j,k], expected)

    def test_emissions(self):
        # intiate relevant classes
        starting = get_base(20, 5, 5, 5)
        emis = get_base(5e6, 5, 5, 5)
        params = ModelParams(emissions = emis, initial = starting, spinup_duration=0.5)
        conc = Concentrations(params, depositon_on = False, advection_on=False, chemistry_on=False)
        # expect one round of emissions from spin up in the constructor class
        expected = 20 + params.time_step * 3600 * 5e6

        self.assertFalse(conc.advected)
        self.assertFalse(conc.chem_applied)
        self.assertTrue(conc.emitted)
        self.assertFalse(conc.deposited)
        self.assertTrue(conc.spun_up)

        for i in range(conc.values['conc_NO2'].shape[0]):
            for j in range(conc.values['conc_NO2'].shape[1]):
                for k in range(conc.values['conc_NO2'].shape[2]):
                    self.assertEqual(conc.values['conc_NO2'][i,j,k], expected)

        # expect two rounds of emissions
        conc.emit()
        expected += params.time_step * 3600 * 5e6

        for i in range(conc.values['conc_NO2'].shape[0]):
            for j in range(conc.values['conc_NO2'].shape[1]):
                for k in range(conc.values['conc_NO2'].shape[2]):
                    self.assertEqual(conc.values['conc_NO2'][i,j,k], expected)

    def test_positive(self):
        # intiate relevant classes
        params = ModelParams(spinup_duration = 0.5)
        conc = Concentrations(params)
        # expect one round of emissions from spin up in the constructor class

        self.assertTrue(conc.advected)
        self.assertTrue(conc.chem_applied)
        self.assertTrue(conc.emitted)
        self.assertTrue(conc.deposited)
        self.assertTrue(conc.spun_up)

        for chemical in conc.chemicals:
            for i in range(conc.values[chemical].shape[0]):
                for j in range(conc.values[chemical].shape[1]):
                    for k in range(conc.values[chemical].shape[2]):
                        self.assertTrue(conc.values[chemical][i,j,k] >= 0)
                        self.assertFalse(conc.values[chemical][i,j,k] is np.nan)

            print('\t {} : ppb \n'.format(chemical))
            print(conc.values[chemical] * 1e9/2.5e19)

if __name__ == '__main__':
    unittest.main()
