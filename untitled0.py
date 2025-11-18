import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def generate_truncated_log_normal(num_points, mu, sigma, lower_bound):
    """
    Generates random numbers from a truncated log-normal distribution.

    Args:
        num_points: The number of random points to generate.
        mu: The mean of the underlying normal distribution.
        sigma: The standard deviation of the underlying normal distribution.
        lower_bound: The lower bound for the underlying normal distribution (Z).

    Returns:
        An array of random numbers from the truncated log-normal distribution.
    """
    # The bounds are defined in terms of standard deviations from the mean
    a = (lower_bound - mu) / sigma
    b = np.inf  # No upper bound

    # Generate the truncated normal variates for Z
    truncated_normal_variates = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=num_points)

    # Exponentiate to get the truncated log-normal data
    return np.exp(truncated_normal_variates)

def estimate_power_law_params(data):
    """
    Estimates the x_min and alpha parameters of a power-law distribution
    from the given data.

    Args:
        data: A numpy array of data points.

    Returns:
        A tuple containing the estimated x_min and alpha.
    """
    x_min = np.min(data)
    n = len(data)
    # Maximum Likelihood Estimator for alpha
    alpha = 1 + n / np.sum(np.log(data / x_min))
    return x_min, alpha

def generate_power_law(n, alpha, x_min):
  """
  Generates random numbers from a power-law distribution using inverse transform sampling.

  Args:
    n: The number of random numbers to generate.
    alpha: The exponent of the power-law distribution.
    x_min: The minimum value of the distribution.

  Returns:
    An array of random numbers from the power-law distribution.
  """
  r = np.random.uniform(0, 1, n)
  return x_min * (1 - r) ** (-1 / (alpha - 1))

def main():
    """
    Generates data, fits a power-law, generates matched data, and visualizes.
    """
    # --- 1. Generate data from a truncated log-normal distribution ---
    num_points = 1000000
    mu = 0.0  # Mean of the underlying normal distribution
    sigma = 1.0  # Standard deviation of the underlying normal distribution
    lower_bound_Z = -3.0  # The lower bound for Z

    truncated_log_normal_data = generate_truncated_log_normal(num_points, mu, sigma, lower_bound_Z)

    # --- 2. Estimate power-law parameters from the generated data ---
    pl_x_min, pl_alpha = estimate_power_law_params(truncated_log_normal_data)

    print("--- Estimated Power-Law Parameters ---")
    print(f"x_min: {pl_x_min:.4f}")
    print(f"alpha: {pl_alpha:.4f}")
    print("--------------------------------------")


    # --- 3. Generate data from the matched power-law distribution ---
    matched_power_law_data = generate_power_law(num_points, pl_alpha, pl_x_min)

    # --- 4. (Optional) Output the first 10 points of each dataset ---
    print("\nFirst 10 points from the truncated log-normal distribution:")
    print(truncated_log_normal_data[:10])
    print("\nFirst 10 points from the matched power-law distribution:")
    print(matched_power_law_data[:10])


    # --- 5. Visualize the distributions for comparison ---
    plt.figure(figsize=(10, 7))

    # Define bins for the histogram on a logarithmic scale
    max_val = max(np.max(truncated_log_normal_data), np.max(matched_power_law_data))
    min_val = min(np.min(truncated_log_normal_data), np.min(matched_power_law_data))
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)

    # Plot histograms on a log-log scale
    plt.hist(truncated_log_normal_data, bins=bins, density=True, alpha=0.7, label='Truncated Log-Normal')
    plt.hist(matched_power_law_data, bins=bins, density=True, alpha=0.7, label='Matched Power-Law')

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Truncated Log-Normal vs. Matched Power-Law Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()