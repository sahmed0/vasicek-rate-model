import numpy as np
import pandas as pd

def simulate_vasicek(r0, a, b, sigma, T=1.0, dt=0.01):
    """
    Simulates a single interest rate path using the Vasicek model.
    
    Parameters:
    r0    : The starting interest rate (e.g., 0.05 for 5%)
    a     : Speed of reversion (how fast it returns to the mean)
    b     : Long-term mean level (where the rate wants to settle)
    sigma : Volatility (how much noise/shake there is)
    T     : Time horizon in years
    dt    : Time step size
    """
    
    # Calculate how many steps are in the simulation based on the time horizon
    N = int(T / dt)
    
    # Create a timeline array so I can plot this easily later
    time = np.linspace(0, T, N)
    
    # Initialising an array to hold my rates, starting with the current rate (r0)
    rates = np.zeros(N)
    rates[0] = r0

    # Euler-Maruyama Discretisation + Monte Carlo Simulation
    # Looping through time to calculate the next rate based on the previous one
    for t in range(1, N):
        # This is the "Drift": pulling the rate back towards the mean (b)
        drift = a * (b - rates[t-1]) * dt
        
        # This is the "Shock": random market noise
        # Using a normal distribution because market moves are generally Gaussian
        shock = sigma * np.sqrt(dt) * np.random.normal()
        
        # The new rate is the old rate + drift + shock
        rates[t] = rates[t-1] + drift + shock
        
    # Returning both the time axis and the rates so the chart is easy to draw
    return time, rates

def calculate_expected_path(r0, a, b, T=1.0, dt=0.01):
    """
    Calculates the theoretical expected value (mean path) of the Vasicek model.
    Noise (sigma) is ignored because E[dWt] = 0.
    """
    N = int(T / dt)
    time = np.linspace(0, T, N)
    
    # Formula: r0 * e^(-at) + b * (1 - e^(-at))
    expected_rates = r0 * np.exp(-a * time) + b * (1 - np.exp(-a * time))
    
    return time, expected_rates

def calculate_yield_curve(r0, a, b, sigma, max_maturity=30):
    """
    Calculates the theoretical Yield Curve based on Vasicek parameters.
    Returns maturities (x-axis) and yields (y-axis).
    """
    maturities = np.linspace(0.1, max_maturity, 100) # From 0.1 years to 30 years
    yields = []

    for tau in maturities:
        # B(tau) formula
        B = (1 - np.exp(-a * tau)) / a
        
        # A(tau) formula
        A_term1 = (b - (sigma**2) / (2 * a**2)) * (B - tau)
        A_term2 = (sigma**2) / (4 * a) * (B**2)
        A = np.exp(A_term1 - A_term2)
        
        # Bond Price P(tau)
        price = A * np.exp(-B * r0)
        
        # Yield = -ln(Price) / tau
        # Avoid division by zero for very small tau
        if tau == 0:
            y = r0
        else:
            y = -np.log(price) / tau
            
        yields.append(y)
        
    return maturities, yields

def calculate_future_distribution(r0, a, b, sigma, t):
    """
    Calculates the statistical distribution of the interest rate at a specific future time 't'.
    Returns the Mean (expected rate) and Standard Deviation (uncertainty).
    """
    
    # 1. The Mean (Expected Value)
    # This is the same formula as the expected path
    expected_mean = r0 * np.exp(-a * t) + b * (1 - np.exp(-a * t))
    
    # 2. The Variance (Derived from the Ito Isometry)
    # Formula: (sigma^2 / 2a) * (1 - e^(-2at))
    variance = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
    
    # Standard Deviation is the square root of variance
    std_dev = np.sqrt(variance)
    
    return expected_mean, std_dev