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
    
    # I need to calculate how many steps are in my simulation based on the time horizon
    N = int(T / dt)
    
    # I'm creating a timeline array so I can plot this easily later
    time = np.linspace(0, T, N)
    
    # I'm initializing an array to hold my rates, starting with the current rate (r0)
    rates = np.zeros(N)
    rates[0] = r0
    
    # Now I'm looping through time to calculate the next rate based on the previous one
    for t in range(1, N):
        # This is the "Drift": pulling the rate back towards the mean (b)
        drift = a * (b - rates[t-1]) * dt
        
        # This is the "Shock": random market noise
        # I'm using a normal distribution because market moves are generally Gaussian
        shock = sigma * np.sqrt(dt) * np.random.normal()
        
        # The new rate is the old rate + drift + shock
        rates[t] = rates[t-1] + drift + shock
        
    # I'm returning both the time axis and the rates so the chart is easy to draw
    return time, rates