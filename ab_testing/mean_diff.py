import numpy as np
from scipy.stats import t

def mean_diff_pvalue_tstat(mu_a, mu_b, var_a, var_b, N_a, N_b,
                           alternative="two-sided", significance=0.05):

    # calculate pool standard error
    pooled_se = np.sqrt(var_a / N_a + var_b / N_b)

    # calculate the t statistic
    delta = mu_a - mu_b
    t_stat = delta / pooled_se

    # calculate degrees of freedom
    dof = N_a + N_b - 2

    # calculate the p-value:
    if alternative == "two-sided":
        p_value = (1.0 - t.cdf(abs(t_stat), dof)) * 2.0
        critical_value = t.ppf(1.0 - significance / 2, dof)
    elif alternative == "larger":
        p_value = 1 - t.cdf(t_stat, dof)
        critical_value = t.ppf(1.0 - significance, dof)
    elif alternative == "smaller":
        p_value = t.cdf(t_stat, dof)
        critical_value = t.ppf(significance, dof)

    return critical_value, t_stat, p_value


def mean_diff_confint(mu_a, mu_b, var_a, var_b, N_a, N_b,
                      alternative="two-sided", significance=0.05):
    # calculate pool standard error
    pooled_se = np.sqrt(var_a / N_a + var_b / N_b)

    # degrees of freedom
    dof = N_a + N_b - 2

    # CI upper and lower bounds
    delta = mu_a - mu_b
    lb = delta - t.ppf(1 - significance / 2, dof) * pooled_se
    ub = delta + t.ppf(1 - significance / 2, dof) * pooled_se
    confint = [lb, ub ] /mu_b

    return confint