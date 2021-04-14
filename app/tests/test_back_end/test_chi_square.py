import numpy as np
import pytest

import app.tests.test_back_end.conftest as cnft
import app.back_end.chi_square as chi_square


class TestChiSquareTestOutput:
    def test_goodness_of_fit(self):
        result = chi_square.test(test_type='goodness-of-fit', alpha=0.05,
                                 n_obs=[[10, 10, 10, 10, 10, 10]])
        assert True

    def test_independence(self):
        res_obs = chi_square.test(test_type='independence', alpha=0.05,
                                  n_obs=[[20, 80], [80, 20]])

        res_exp = {
            'n': 200, 'dof': 1,
            'n_exp': np.array([[50, 50], [50, 50]]),
            'p_exp': np.array([[0.25, 0.25], [0.25, 0.25]]),
            'lambda_factor': 0.36,
            'qchi2': 3.8415, 'chi2_stat': 72,
            'p_value': 0, 'power': 1
        }

        cnft.assert_equal(res_exp, res_obs)

    def test_homogenity(self):
        res_obs = chi_square.test(test_type='homogenity', alpha=0.05,
                                  n_obs=[[20, 80], [80, 20]])

        res_exp = {
            'n': 200, 'dof': 1,
            'n_exp': np.array([[50, 50], [50, 50]]),
            'p_exp': np.array([[0.5, 0.5], [0.5, 0.5]]),
            'lambda_factor': 0.36,
            'qchi2': 3.8415, 'chi2_stat': 72,
            'p_value': 0, 'power': 1
        }

        cnft.assert_equal(res_exp, res_obs)
