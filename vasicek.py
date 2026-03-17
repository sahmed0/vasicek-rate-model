import numpy as np

def simulate_vasicek(r0, a, b, sigma, T=1.0, dt=0.01, num_sims=1):
    """
    Simulate interest rate paths using the Vasicek model.
    
    Uses a vectorized Euler-Maruyama discretisation to generate multiple paths simultaneously.

    :param r0: Initial interest rate (short rate).
    :param a: Speed of mean reversion.
    :param b: Long-term mean level.
    :param sigma: Volatility of the rate process.
    :param T: Total time horizon in years.
    :param dt: Time step size for discretisation.
    :param num_sims: Number of independent simulation paths.
    :return: Tuple containing (time_axis, rate_paths_matrix).
    """
    N = int(T / dt)
    time = np.linspace(0, T, N)
    
    rates = np.zeros((N, num_sims))
    rates[0, :] = r0

    # Scale random shocks by sqrt(dt) as variance grows linearly with time.
    shocks = sigma * np.sqrt(dt) * np.random.normal(size=(N, num_sims))
    
    # Iterate through time steps to apply the Euler-Maruyama update rule.
    for t in range(1, N):
        drift = a * (b - rates[t-1, :]) * dt
        rates[t, :] = rates[t-1, :] + drift + shocks[t, :]
        
    return time, rates

def calculate_expected_path(r0, a, b, T=1.0, dt=0.01):
    """
    Calculate the theoretical expected value (mean path) of the Vasicek model.
    
    Ignores the stochastic component as the expectation of a Wiener process is zero.

    :param r0: Initial interest rate.
    :param a: Speed of mean reversion.
    :param b: Long-term mean level.
    :param T: Total time horizon in years.
    :param dt: Time step size for calculation.
    :return: Tuple containing (time_axis, expected_rates).
    """
    N = int(T / dt)
    time = np.linspace(0, T, N)
    
    # E[r_t] = r0 * e^(-at) + b * (1 - e^(-at))
    expected_rates = r0 * np.exp(-a * time) + b * (1 - np.exp(-a * time))
    
    return time, expected_rates

def calculate_yield_curve(r0, a, b, sigma, max_maturity=30):
    """
    Calculate the theoretical Yield Curve based on Vasicek parameters.
    
    Derives the yield for zero-coupon bonds using the affine term structure property.

    :param r0: Initial short rate.
    :param a: Speed of mean reversion.
    :param b: Long-term mean level.
    :param sigma: Volatility.
    :param max_maturity: Maximum maturity for the curve in years.
    :return: Tuple containing (maturities, yields).
    """
    maturities = np.linspace(0.1, max_maturity, 100)

    B = (1 - np.exp(-a * maturities)) / a
    
    A_term1 = (b - (sigma**2) / (2 * a**2)) * (B - maturities)
    A_term2 = (sigma**2) / (4 * a) * (B**2)

    ln_A = A_term1 - A_term2
    ln_price = ln_A - (B * r0)
    
    # Yield = -ln(Price) / maturity
    yields = np.zeros_like(maturities)
    nonzero_idx = maturities > 0
    yields[nonzero_idx] = -ln_price[nonzero_idx] / maturities[nonzero_idx]
    
    if not nonzero_idx[0]:
        yields[0] = r0
        
    return maturities, yields

def calculate_future_distribution(r0, a, b, sigma, t):
    """
    Calculate the statistical distribution of the interest rate at a specific future time.
    
    Computes the mean and standard deviation derived from the Ornstein-Uhlenbeck transition density.

    :param r0: Initial interest rate.
    :param a: Speed of mean reversion.
    :param b: Long-term mean level.
    :param sigma: Volatility.
    :param t: Forecast horizon in years.
    :return: Tuple containing (expected_mean, std_dev).
    """
    
    expected_mean = r0 * np.exp(-a * t) + b * (1 - np.exp(-a * t))
    
    # Variance (derived from Ito Isometry)
    variance = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
    
    std_dev = np.sqrt(variance)
    
    return expected_mean, std_dev
