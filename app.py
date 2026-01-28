import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from vasicek import simulate_vasicek  # I'm importing the logic I wrote in the other file

# --- CONFIGURATION ---
st.set_page_config(page_title="Yield Curve & Vasicek Model", layout="wide")

st.title("📉 Vasicek Interest Rate Simulator")
st.markdown("""
This tool simulates future interest rate paths using the **Vasicek Model**.
It assumes rates are mean-reverting—they fluctuate but eventually get pulled back to a long-term average.
""")

# --- SIDEBAR (CONTROLS) ---
st.sidebar.header("Model Parameters")

# I'm adding sliders so the user can play with the inputs
current_rate = st.sidebar.slider("Current Interest Rate (r0)", 0.0, 0.1, 0.04, 0.001)
long_term_mean = st.sidebar.slider("Long-Term Mean (b)", 0.0, 0.1, 0.05, 0.001)
reversion_speed = st.sidebar.slider("Reversion Speed (a)", 0.0, 2.0, 0.5, 0.1)
volatility = st.sidebar.slider("Volatility (sigma)", 0.0, 0.1, 0.02, 0.001)
time_horizon = st.sidebar.slider("Time Horizon (Years)", 1, 30, 10)

num_simulations = st.sidebar.slider("Number of Simulations", 1, 50, 10)

# --- RUNNING THE SIMULATION ---

# I'm setting up the plot figure
fig = go.Figure()

# I'm running the simulation multiple times to show different possible futures
for i in range(num_simulations):
    # Calling my function from vasicek.py
    time_axis, rate_path = simulate_vasicek(
        r0=current_rate,
        a=reversion_speed,
        b=long_term_mean,
        sigma=volatility,
        T=time_horizon
    )
    
    # Adding each path to the chart
    # I'm setting opacity to 0.5 so it looks like a 'cloud' of possibilities
    fig.add_trace(go.Scatter(x=time_axis, y=rate_path, mode='lines', opacity=0.5, name=f'Sim {i+1}'))

# --- DISPLAYING THE RESULTS ---

# I'm adding a line for the Long-Term Mean so it's clear where the rates are heading
fig.add_hline(y=long_term_mean, line_dash="dash", line_color="red", annotation_text="Long-Term Mean")

fig.update_layout(
    title=f"Projected Interest Rate Paths over {time_horizon} Years",
    xaxis_title="Time (Years)",
    yaxis_title="Interest Rate",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# --- EXPLANATION FOR RECRUITERS ---
st.info("""
**Quant Note:** This model uses the Euler-Maruyama method to discretize the Vasicek SDE.
Notice how high 'Reversion Speed' pulls rates to the red line faster, while high 'Volatility' makes the paths more jagged.
""")