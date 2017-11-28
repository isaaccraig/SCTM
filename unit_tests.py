
 import unittest


get_base = lambda val, xdim, ydim, zdim: {      'conc_NO2' : val * np.ones([xdim, ydim, zdim]),
                                                'conc_APN' : val * np.ones([xdim, ydim, zdim]),
                                                'conc_AP'  : val * np.ones([xdim, ydim, zdim]),
                                                'conc_NO'  : val * np.ones([xdim, ydim, zdim]),
                                                'conc_O3'  : val * np.ones([xdim, ydim, zdim]),
                                                'conc_HNO3': val * np.ones([xdim, ydim, zdim]),
                                                'conc_HO'  : val * np.ones([xdim, ydim, zdim]),
                                                'conc_HO2' : val * np.ones([xdim, ydim, zdim]),
                                                'conc_PROD': val * np.ones([xdim, ydim, zdim])}

class TestParameters(unittest.TestCase):
    def test_hour_generator(self):
        params = ModelParams()
        hours = params.hour_generator()
        hours_collect = []
        for i in range(3):
            hours_collect[i] = next(hours)
        self.assertEqual(hours_collect, params.time_range[:3])

class TestConcentrationClass(unittest.TestCase):
    def test_emissions(self):
        # intiate relevant classes
        starting = get_base(20, 5, 5, 5)
        emis = get_base(10, 5, 5, 5)
        params = ModelParams(emissions = emis, initial = starting)
        conc = Concentrations(params)
        # expect one round of emissions from spin up in the constructor class
        expected = starting['conc_NO2'] + self.params.time_step * (1e5)**2 * 1/self.params.height * 6.022e23) * emis['conc_NO2']
        self.assertEqual(conc.values['conc_NO2'], expected)
        # expect two rounds of emissions
        self.emit()
        expected += self.params.time_step * (1e5)**2 * 1/self.params.height * 6.022e23) * emis['conc_NO2']
        self.assertEqual(conc.values['conc_NO2'], expected)







if __name__ == '__main__':
    unittest.main()
