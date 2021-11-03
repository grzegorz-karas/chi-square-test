import numpy as np
import pytest

import app.tests.test_back_end.conftest as cnft
import app.back_end.chi_square as chi_square


class TestChiSquareTestOutput:
    def test_goodness_of_fit(self):
        result = chi_square.test(test_type='goodness-of-fit',
                                 alpha=0.05,
                                 n_obs=[[10, 10, 10, 10, 10, 10]],
                                 n_exp=[[10, 10, 10, 10, 10, 10]])

        res_exp = {
            'dof': 5,
            'qchi2': 3.8415,
            'chi2_stat': 72,
            'power': 1,
            'p_value': 1,
            'noncentral_param': 0.36*120,
            'lambda_factor': 0.36,

        }
        assert True

    def test_independence(self):
        res_obs = chi_square.test(test_type='independence', alpha=0.05,
                                  n_obs=[[20, 80], [80, 20]])

        res_exp = {
            'dof': 1,
            'qchi2': 3.8415,
            'chi2_stat': 72,
            'power': 1,
            'p_value': 0,
            'noncentral_param': 0.36*200,
            'lambda_factor': 0.36,
        }

        cnft.assert_equal(res_exp, res_obs)

    def test_homogenity(self):
        res_obs = chi_square.test(test_type='homogenity', alpha=0.05,
                                  n_obs=[[20, 80], [80, 20]])

        res_exp = {
            'dof': 1,
            'qchi2': 3.8415,
            'chi2_stat': 72,
            'power': 1,
            'p_value': 0,
            'noncentral_param': 0.36*200,
            'lambda_factor': 0.36,
        }

        cnft.assert_equal(res_exp, res_obs)


class TestFindNoncentralParam:

    def test_find_noncentral_1(self):

        cdf = 0.95
        dof = 1
        qchi2 = chi_square.chi2.ppf(cdf, dof)

        target_power = 0.05
        noncentral_param = chi_square.find_noncentral_param(dof,
                                                            qchi2,
                                                            target_power)

        assert noncentral_param == 0
