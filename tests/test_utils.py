import unittest
import random
import string
import statsmodels.api as sm
from src.robustify.utils import simple_ols
from src.robustify.utils import space_size


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

    def test_p_less_one(self):
        '''
        Test pvalues less than 1
        '''
        n = 7
        actual_p = self.actual['p'].flatten().round(n).tolist()
        for p in actual_p:
            self.assertLess(p, 1)

    def test_aic(self):
        '''
        Test aic againts statmodels
        '''
        actual_aic = self.actual['aic'].item(0)
        target_aic = self.target.aic
        self.assertAlmostEqual(actual_aic, target_aic)

    def test_bic(self):
        '''
        Test bic againts statmodels
        '''
        actual_bic = self.actual['bic'].item(0)
        target_bic = self.target.bic
        self.assertAlmostEqual(actual_bic, target_bic)


class TestSpaceSize(unittest.TestCase):

    def test_output_type(self):
        '''
        Test output type of space_size()
        '''
        int_list = [random.randrange(10) for i in range(10)]
        float_list = [random.uniform(0, 1) for i in range(10)]
        str_list = [random.choice(string.ascii_lowercase) for i in range(10)]
        self.assertIs(type(space_size(int_list)), int)
        self.assertIs(type(space_size(float_list)), int)
        self.assertIs(type(space_size(str_list)), int)


if __name__ == '__main__':
    unittest.main()
