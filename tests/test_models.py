import unittest
import statsmodels.api as sm
import numpy as np
from src.robustify.models import OLSRobust


class TestOLSRobust(unittest.TestCase):

    def setUp(self):
        self.data = sm.datasets.randhie.load_pandas()
        self.sample = self.data.data.sample(1000)
        self.y = self.sample.iloc[:, :1]
        self.x = self.sample.iloc[:, 1:2]
        self.c = self.sample.iloc[:, 2:]
        self.actual = OLSRobust(y=self.y, x=self.x)

    def test_estimate_type(self):
        '''
        Test _estimate() method returned types.
        '''
        b, p, aic, bic = self.actual._estimate(y=self.y,
                                               x=self.x,
                                               mode='simple')
        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(p, np.ndarray)
        self.assertIsInstance(aic, np.ndarray)
        self.assertIsInstance(bic, np.ndarray)

    def test_strap_type(self):
        '''
        Test _strap() method returned types.
        '''
        b, p, aic, bic = self.actual._strap(comb_var=self.sample,
                                            mode='simple',
                                            sample_size=1000,
                                            replace=True)
        self.assertIsInstance(b, float)
        self.assertIsInstance(p, float)
        self.assertIsInstance(aic, float)
        self.assertIsInstance(bic, float)


if __name__ == '__main__':
    unittest.main()
