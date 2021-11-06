from math import ceil
import numpy as np
from scipy.stats import chi2, ncx2, chi2_contingency, chisquare
from scipy.stats.contingency import expected_freq
from typing import List, Dict, Any, Optional

Matrix = List[List[float]]


def cdf(x: float, dof: int, noncentral_param: Optional[float]) -> float:

    if noncentral_param and noncentral_param > 0:
        return ncx2.cdf(x, dof, noncentral_param)
    else:
        return chi2.cdf(x, dof)


def convert_to_2D(m: Matrix) -> Matrix:

    m = np.array(m)

    if len(m.shape) == 1:
        m = np.reshape(m, (1, -1))

    return m


def test(test_type: str,
         alpha: float,
         n_obs: Matrix,
         n_exp: Optional[Matrix] = None,
         ) -> Dict:

    n_obs = convert_to_2D(n_obs)

    n = n_obs.sum()

    if test_type == 'homogenity':

        n_obs_i__ = n_obs.sum(axis=1, keepdims=True)
        n_obs___j = n_obs.sum(axis=0, keepdims=True)

        p_obs = n_obs / n_obs_i__

        p_obs_i__ = n_obs_i__ / n
        p_obs___j = n_obs___j / n

        chi2_stat, p_value, dof, _ = chi2_contingency(n_obs, correction=False)

        delta = p_obs - p_obs___j

        if np.isclose(delta.sum(axis=1), np.zeros(delta.shape[0])).all():
            lambda_factor = (
                1/p_obs___j * (
                    (delta**2 * p_obs_i__).sum(axis=0, keepdims=True)
                    - (delta * p_obs_i__).sum(axis=0, keepdims=True)**2
                )
            ).sum()
        else:
            lambda_factor = None

    else:

        p_obs = n_obs / n

        if test_type == 'goodness-of-fit':

            if n_exp:
                n_exp = convert_to_2D(n_exp)
            else:
                raise ValueError("variable n_exp not provided")

            p_exp = n_exp / n

            k = n_obs.shape[-1]
            dof = k - 1
            chi2_stat, p_value = chisquare(np.ravel(n_obs), np.ravel(n_exp))

            lambda_factor = ((p_obs - p_exp)**2 / p_exp).sum()

        elif test_type == 'independence':

            p_obs_i__ = p_obs.sum(axis=1, keepdims=True)
            p_obs___j = p_obs.sum(axis=0, keepdims=True)

            n_exp = expected_freq(n_obs)

            p_exp = n_exp / n

            chi2_stat, p_value, dof, _ = chi2_contingency(
                n_obs, correction=False)

            delta = p_obs - p_exp

            delta_i__ = delta.sum(axis=1, keepdims=True)
            delta___j = delta.sum(axis=0, keepdims=True)

            if np.isclose(delta.sum(), 0):
                lambda_factor = (
                    (delta**2 / p_exp).sum() +
                    (delta_i__**2 / p_obs_i__).sum() +
                    (delta___j**2 / p_obs___j).sum()
                )
            else:
                lambda_factor = None

    qchi2 = chi2.ppf(1-alpha, dof)

    noncentral_param = n * lambda_factor

    if noncentral_param >= 0:
        power = 1 - cdf(qchi2, dof, noncentral_param)
    else:
        power = None

    result = {
        'dof': dof,
        'qchi2': qchi2,
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'power': power,
        'noncentral_param': noncentral_param,
        'lambda_factor': lambda_factor,
    }
    return result


def find_noncentral_param(dof, qchi2, target_power):

    noncentral_param = 0
    step = 0.01
    power = 0

    while True:
        power = ncx2.sf(qchi2, dof, noncentral_param)

        if np.round(target_power, 4) <= np.round(power, 4):
            return noncentral_param

        noncentral_param += step


def sample_size(dof: int,
                lambda_factor: float,
                qchi2: float,
                target_power: float
                ) -> int:

    if lambda_factor == 0:
        return 0

    noncentral_param = find_noncentral_param(dof, qchi2, target_power)

    n = ceil(noncentral_param / lambda_factor)

    return n


def test_and_sample_size(test_type: str,
                         alpha: float,
                         target_power: float,
                         n_obs: Matrix,
                         n_exp: Optional[Matrix] = None
                         ) -> Dict:

    test_output = test(test_type, alpha, n_obs, n_exp)

    target_power_sample_size = sample_size(
        dof=test_output["dof"],
        lambda_factor=test_output["lambda_factor"],
        qchi2=test_output["qchi2"],
        target_power=target_power
    )

    test_output.update({
        "target_power": target_power,
        "target_power_sample_size": target_power_sample_size
    })

    return test_output
