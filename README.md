# Chi-square-test (chi-squared-test)

The $\chi^2$-test is a statistical method to test hypothesis, where random variable
follows multinomial distribution. It tests a null hypothesis stating that the frequency distribution of certain
events is consistent with a particular theoretical distribution.

There are three types of hypothesis testing

1. Goodness of fit.
2. Homogenity.
3. Independence.

Please read details in the [manuscript.pdf](app/back_end/manuscript/doc.pdf).

# Content description

The module [chi_square.py](app/back_end/chi_sqaure.py) consist of functions that help to:

1. Calculate cumulative $\chi^2$ distribution with or without noncentral parameter.
2. Perform the $\chi^2$ test for all three types of hypothesis.
3. Find noncentral parameter for defined $\chi^2$ quantile and target power.
4. Calculate sample size for given target power.
5. Perform the $\chi^2$ test and calculate the sample size for given target power.

Please see the [test folder](app/tests/test_back_end/test_chi_square.py) for checking how the functions shall be called.

# Disclaimer

Code was written on the basis of the following article

Guenther, W. (1977).
Power and Sample Size for Approximate Chi-Square Tests.
The American Statistician, 31(2), 83-85.
