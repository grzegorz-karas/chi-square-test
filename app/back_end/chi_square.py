from collections import namedtuple
import numpy as np
from scipy.stats import chi2, ncx2, chi2_contingency, chisquare
from typing import List, Dict, Any, Optional

Matrix = List[List[float]]


def cdf(x: float, dof: int, noncentral_param: Optional[float]) -> float:

    if noncentral_param and noncentral_param > 0:
        return ncx2.cdf(x, dof, noncentral_param)
    else:
        return chi2.cdf(x, dof)


def unify_input(opt_input, opt, n, if_missing):

    if not opt_input:
        n_opt_input = if_missing
        p_opt_input = n_opt_input / n
    elif opt == 'cnt':
        n_opt_input = opt_input
        p_opt_input = opt_input / n
    elif opt == 'prob':
        n_opt_input = round(opt_input * n)
        p_opt_input = opt_input

    return n_opt_input, p_opt_input


def test(test_type: str,
         alpha: float,
         n_obs: Matrix,
         opt: Optional[str] = None,
         exp: Optional[Matrix] = None,
         alt: Optional[Matrix] = None,
         ) -> Dict:

    n_obs = np.array(n_obs)
    exp = np.array(exp)
    alt = np.array(alt)

    n = n_obs.sum()

    p_obs = n_obs / n

    if test_type == 'goodness-of-fit':
        n_obs_vector = n_obs[0]
        k = len(n_obs_vector)

        n_exp, p_exp = unify_input(exp, opt, n, np.array([[1/k * n] * k]))

        n_alt, p_alt = unify_input(alt, opt, n, n_obs)

        n_exp_vector = n_exp[0]

        chi2_stat, p_value = chisquare(n_obs_vector, n_exp_vector)
        dof = k - 1

        lambda_factor = ((p_alt - p_exp)**2 / p_exp).sum()

    elif test_type == 'independence':

        chi2_stat, p_value, dof, n_exp = chi2_contingency(
            n_obs, correction=False)

        p_obs_i__ = p_obs.sum(axis=1, keepdims=True)
        p_obs___j = p_obs.sum(axis=0, keepdims=True)

        print(f'exp input variable in the test of {test_type} is ignored')
        p_exp = n_exp / n

        n_alt, p_alt = unify_input(alt, opt, n, n_obs)

        delta = p_alt - p_exp
        delta_i__ = delta.sum(axis=1, keepdims=True)
        delta___j = delta.sum(axis=0, keepdims=True)

        if round(delta.sum(), 4) == 0:
            lambda_factor = (
                (delta**2 / p_exp).sum() +
                (delta_i__**2 / p_obs_i__).sum() +
                (delta___j**2 / p_obs___j).sum()
            )
        else:
            lambda_factor = None

    elif test_type == 'homogenity':

        chi2_stat, p_value, dof, n_exp = chi2_contingency(
            n_obs, correction=False)

        n_obs_i__ = n_obs.sum(axis=1, keepdims=True)
        n_obs___j = n_obs.sum(axis=0, keepdims=True)

        p_obs_i__ = p_obs.sum(axis=1, keepdims=True)
        p_obs___j = p_obs.sum(axis=0, keepdims=True)

        _, p_exp = unify_input(exp, opt, n, n_obs___j)

        n_alt, p_alt = unify_input(alt, opt, n_obs_i__, n_obs)

        delta = p_alt - p_exp

        if np.isclose(delta.sum(axis=1), np.zeros(delta.shape[0])).all():
            lambda_factor = (
                1/p_exp * (
                    (delta**2 * p_obs_i__).sum(axis=0, keepdims=True)
                    - (delta * p_obs_i__).sum(axis=0, keepdims=True)**2
                )
            ).sum()
        else:
            lambda_factor = None

    else:
        raise ValueError(
            '''test_type not in allowed values ''' +
            '''["goodness-of-fit", "independence", "homogenity"]'''
        )

    qchi2 = chi2.ppf(1-alpha, dof)
    noncentral_param = n * lambda_factor

    if lambda_factor > 0:
        power = 1 - cdf(qchi2, dof, noncentral_param)
    else:
        power = None

    result = {
        'n': n, 'dof': dof,
        'n_exp': n_exp, 'p_exp': p_exp,
        'lambda_factor': lambda_factor,
        'qchi2': qchi2, 'chi2_stat': chi2_stat,
        'p_value': p_value, 'power': power
    }
    return result


def sample_size(dof: int,
                lambda_factor: float,
                qchi2: float,
                target_power: float
                ) -> int:

    noncentral_param = find_noncentral_param(dof, qchi2, target_power)

    n = int(lambda_factor / noncentral_param)

    return n
