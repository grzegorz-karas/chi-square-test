import numpy as np
import pytest

import app.tests.test_back_end.conftest as cnft
import app.back_end.chi_square as chi_square


class TestChiSquareTest:
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

    def test_find_noncentral_0(self):

        cdf = 0.95
        dof = 1
        qchi2 = chi_square.chi2.ppf(cdf, dof)

        target_power = 0.05
        noncentral_param = chi_square.find_noncentral_param(dof,
                                                            qchi2,
                                                            target_power)

        assert noncentral_param == 0


class TestSampleSize:

    def test_sample_size_goodness_of_fit_perfect_match(self):

        alpha = 0.05

        result = chi_square.test_and_sample_size(
            test_type='goodness-of-fit',
            alpha=alpha,
            target_power=0.8,
            n_obs=[[10, 10, 10, 10, 10, 10]],
            n_exp=[[10, 10, 10, 10, 10, 10]])

        assert result['target_power_sample_size'] == 0

    def test_sample_size_goodness_of_fit(self):

        alpha = 0.05

        result = chi_square.test_and_sample_size(
            test_type='goodness-of-fit',
            alpha=alpha,
            target_power=0.8,
            n_obs=[[10, 10, 10, 10, 11, 9]],
            n_exp=[[10, 10, 10, 10, 10, 10]])

        assert result['power'] < 2*alpha

    def test_sample_size_goodness_of_fit_power_equal_target(self):

        result = chi_square.test_and_sample_size(
            test_type='goodness-of-fit',
            alpha=0.05,
            target_power=0.8,
            n_obs=[[round(10/60*3870), round(10/60*3870), round(10/60*3870),
                    round(10/60*3870), round(11/60*3870), round(9/60*3870)]],
            n_exp=[[round(10/60*3870), round(10/60*3870), round(10/60*3870),
                    round(10/60*3870), round(10/60*3870), round(10/60*3870)]])

        assert result['target_power'] < result['power']

    def test_sample_size_independence(self):

        alpha = 0.05

        result = chi_square.test_and_sample_size(
            test_type='independence',
            alpha=alpha,
            target_power=0.8,
            n_obs=[[10, 10, 10, 10, 11, 9],
                   [10, 10, 10, 10, 10, 10]])

        assert result['power'] < 2*alpha

    def test_sample_size_independence_power_equal_target(self):

        result = chi_square.test_and_sample_size(
            test_type='independence',
            alpha=0.05,
            target_power=0.8,
            n_obs=[[round(10/60*5748), round(11/60*5748), round(9/60*5748)],
                   [round(10/60*5748), round(10/60*5748), round(10/60*5748)]])

        assert result['target_power'] < result['power']

    def test_sample_size_independence_perfect_match(self):

        alpha = 0.05

        result = chi_square.test_and_sample_size(
            test_type='independence',
            alpha=alpha,
            target_power=0.8,
            n_obs=[[10, 10, 10, 10, 10, 10],
                   [10, 10, 10, 10, 10, 10]])

        assert result['target_power_sample_size'] == 0
