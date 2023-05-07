import numpy as np
from math import sqrt
from scipy import stats
from scipy.stats import t
from statsmodels.stats.proportion import proportions_ztest
from tqdm.notebook import tqdm


def prop_diff_p_value_zstat(sample_success_a, sample_success_b, sample_size_a,
                            sample_size_b, alternative="two-sided"):

    successes = np.array([sample_success_a, sample_success_b])
    samples = np.array([sample_size_a, sample_size_b])

    stat, p_value = proportions_ztest(count=successes, nobs=samples, alternative=alternative)

    return stat, p_value

def prop_diff_confint(sample_success_a, sample_success_b, sample_size_a,
                    sample_size_b, significance=0.05):

    prop_a = sample_success_a / sample_size_a
    prop_b = sample_success_b / sample_size_b
    var = prop_a * (1 - prop_a) / sample_size_a + prop_b * (1 - prop_b) / sample_size_b
    se = np.sqrt(var)
    z = stats.norm(loc=0, scale=1).ppf(1 - significance / 2)
    prop_diff = prop_a - prop_b
    confint = prop_diff + np.array([-1, 1]) * z * se
    confint = confint/ prop_b

    percent_diff = np.round((prop_diff / prop_b) * 100, 2)

    return prop_diff, confint, z

def test_prop_diff(df, numerator, denominator, significance=0.05,
                 alternative="two-sided", report=True):

    # Document hypothesis test settings
    results = {
        "proportion parameter": f"p = {numerator}/{denominator}",
        "significance": significance,
        "alternative": alternative,
        "null_hypothesis": "Ho (null hypothesis) : p_test = p_control",
    }
    # Alternative hypothesis
    if alternative == "larger":
        results["alternative_hypothesis"] = "Ha (alternative hypothesis) : p_test > p_control"
    elif alternative == "smaller":
        results["alternative_hypothesis"] = "Ha (alternative hypothesis) : p_test < p_control"
    else:
        results["alternative_hypothesis"] = "Ha (alternative hypothesis) : p_test != p_control"

    # Get inputs
    sample_success_a, sample_size_a = (df[numerator].test, df[denominator].test)
    sample_success_b, sample_size_b = (df[numerator].control, df[denominator].control)

    # Compute z-stat and p-value
    results["z_stat"], results["p_value"] = prop_diff_p_value_zstat(
        sample_success_a, sample_success_b, sample_size_a, sample_size_b, alternative
    )

    # Calcualte CI
    (results["prop_diff"], results["confint"], results["Critical value"]) = prop_diff_confint(
        sample_success_a, sample_success_b, sample_size_a, sample_size_b, significance
    )

    #Process results
    results["relative_diff"] = np.round((results["prop_diff"] / (sample_success_b / sample_size_b)), 2)

    results["Conclusion"] = (
        "=> Fail to reject the null hypothesis"
        if results["p_value"] > significance
        else "=> Reject the null hypothesis - evidence suggests the alternative hypothesis is true"
    )

    if results["p_value"] < 0.01:
        results["max_confidence_level"] = "***"
    elif results["p_value"] < 0.05:
        results["max_confidence_level"] = "**"
    elif results["p_value"] < 0.10:
        results["max_confidence_level"] = "*"
    else:
        results["max_confidence_level"] = "."

    # Report
    if report:
        print(f"Z test for difference in proportions for: p = {numerator}/{denominator} \n")
        print(results["null_hypothesis"])
        print(results["alternative_hypothesis"])
        print("\n")
        print("Results:\n")
        print("z_stat: %0.3f, p_value: %0.3f" % (results["z_stat"], results["p_value"]))
        print("Critical value: %0.3f" % (results["Critical value"]))
        print("\n")
        print(results["Conclusion"])
        print("\n")
        print(
            "Observed diff in proportions: %0.3f , relative diff %0.3f"
            % (results["prop_diff"], results["relative_diff"])
        )
        print("95 percent confidence interval (relative change): [ %0.3f , %0.3f ]" % (
            results["confint"][0],
             results["confint"][1] )
             )

    return results