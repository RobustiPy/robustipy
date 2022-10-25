import unittest
import statsmodels.api as sm
from src.python.utils import simple_ols


class TestSimpleOLS(unittest.TestCase):

    def setUp(self):
        self.data = sm.datasets.randhie.load_pandas()
        self.sample = self.data.data.sample(1000)
        self.y = self.sample.iloc[:, :1]
        self.x = self.sample.iloc[:, 1:]
        self.actual = simple_ols(self.y, self.x)
        self.target = sm.OLS(self.y, self.x, hasconst=False).fit()

    def test_beta(self):
        '''
        Test estimates against statsmodels
        up to n (default to 7) decimal places.
        '''
        n = 7
        actual_b = self.actual['b'].flatten().round(n).tolist()
        target_b = self.target.params.round(n).to_numpy().tolist()
        self.assertListEqual(actual_b, target_b)

    def test_pvalues(self):
        '''
        Test pvalues against statsmodels
        up to n (default to 7) decimal places.
        '''
        n = 7
        actual_p = self.actual['p'].flatten().round(n).tolist()
        target_p = self.target.pvalues.round(n).to_numpy().tolist()
        self.assertListEqual(actual_p, target_p)


if __name__ == '__main__':
    unittest.main()
