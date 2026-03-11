import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

def create_simulation_chart(time_axis, rate_paths, expected_path, num_sims, long_term_mean, time_horizon):
    """
    Create the main Vasicek rate simulation chart.
    
    :param time_axis: 1D flat array for the x-axis time series.
    :param rate_paths: 2D Matrix of dimension (N_steps, num_sims) containing generated paths.
    :param expected_path: Tuple (exp_time, exp_rates) from expected path calculation.
    :param num_sims: Number of simulations for naming traces.
    :param long_term_mean: Reversion mean (b parameter).
    :param time_horizon: Simulation length in years (T parameter).
    :return: go.Figure: The configured Plotly figure.
    """
    fig = go.Figure()
    
    for i in range(num_sims):
        fig.add_trace(go.Scatter(
            x=time_axis, y=rate_paths[:, i], 
            mode='lines', 
            opacity=0.9, 
            line=dict(width=1),
            name=f'Sim {i+1}',
            showlegend=False
        ))

    exp_time, exp_rates = expected_path
    fig.add_trace(go.Scatter(
        x=exp_time, y=exp_rates, 
        mode='lines', 
        name='Expected Path',
        line=dict(color='#111827', width=3, dash='dash')
    ))
    
    fig.add_hline(y=long_term_mean, line_dash="dot", line_color="#DC2626", annotation_text="Long Term Mean")

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Time (Years)",
        yaxis_title="Interest Rate",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit", size=14, color="#4B5563"),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=14, font_family="Outfit")
    )
    
    return fig

def create_yield_curve_chart(maturities, yields):
    """
    Create the implied yield curve chart based on Vasicek parameters.
    
    :param maturities: X-axis data points (time in years).
    :param yields: Y-axis data points (implied zeros yield).
    :return: go.Figure: The configured Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maturities, y=yields, 
        mode='lines', 
        name='Yield Curve',
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)',
        line=dict(color='#2563EB', width=4)
    ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Maturity (Years)",
        yaxis_title="Yield",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit", size=14, color="#4B5563"),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=14, font_family="Outfit")
    )
    
    return fig

def create_distribution_chart(mu, std, x_axis, y_axis):
    """
    Create a probability distribution chart showing future rate forecast distribution.
    
    :param mu: Expected rate mean.
    :param std: Expected rate standard deviation.
    :param x_axis: X-axis plot values (rate levels).
    :param y_axis: Y-axis plot values (probability densities).
    :return: go.Figure: The configured Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_axis,
        mode='lines',
        name='Probability Density',
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)',
        line=dict(color="#2563EB", width=3)
    ))
    
    fig.add_vline(x=mu, line_dash="dash", line_color="#111827")
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Interest Rate",
        yaxis_title="Probability Density",
        xaxis=dict(tickformat=".1%"),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit", size=14, color="#4B5563"),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=14, font_family="Outfit")
    )
    
    return fig
