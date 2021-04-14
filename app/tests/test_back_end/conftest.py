import numbers as n
import numpy as np
import pytest


def rounded(number, digits=4):
    return round(number, digits)


def is_number(num):
    return isinstance(num, n.Number)


def is_array(array):
    return isinstance(array, np.ndarray)


def assert_equal(res_exp, res_out):

    for key, res_out_value in res_out.items():
        if is_number(res_out_value):
            assert rounded(res_out_value) == rounded(res_exp[key])
        elif is_array(res_out_value):
            assert (res_out_value == res_exp[key]).all()
        else:
            assert res_out_value == res_exp[key]
