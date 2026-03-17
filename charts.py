import numpy as np
import colorsys

def get_vivid_color(index, total):
    """
    Returns a vivid color from the full HSV spectrum.
    """
    if total <= 1:
        hue = 0.0
    else:
        hue = index / total
    
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def create_simulation_chart(time_axis, rate_paths, expected_path, num_sims, long_term_mean, time_horizon):
    """
    Create the main Vasicek rate simulation chart with vivid multicolor paths.
    """
    data = []
    
    # Simulation paths
    for i in range(num_sims):
        data.append({
            "x": time_axis.tolist() if isinstance(time_axis, np.ndarray) else time_axis,
            "y": rate_paths[:, i].tolist() if isinstance(rate_paths, np.ndarray) else rate_paths[:, i],
            "mode": "lines",
            "opacity": 0.4,
            "line": {"width": 1, "color": get_vivid_color(i, num_sims)},
            "name": f"Sim {i+1}",
            "showlegend": False,
            "hoverinfo": "none"
        })

    # Expected Rate Line
    exp_time, exp_rates = expected_path
    data.append({
        "x": exp_time.tolist() if isinstance(exp_time, np.ndarray) else exp_time,
        "y": exp_rates.tolist() if isinstance(exp_rates, np.ndarray) else exp_rates,
        "mode": "lines",
        "name": "Expected Rate",
        "showlegend": True,
        "line": {"color": "#1a1a1a", "width": 3, "dash": "dot"}
    })

    # Long Term Mean Line
    data.append({
        "x": [0, time_horizon],
        "y": [long_term_mean, long_term_mean],
        "mode": "lines",
        "name": "Long Term Mean",
        "showlegend": True,
        "line": {"color": "#000000", "width": 2}
    })

    layout = {
        "template": "plotly_white",
        "xaxis": {
            "title": {"text": "Years", "standoff": 20},
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linecolor": "#e9ecef",
            "linewidth": 2,
            "automargin": True
        },
        "yaxis": {
            "title": {"text": "Interest Rate", "standoff": 20},
            "showgrid": True,
            "gridcolor": "#f0f2f5",
            "zeroline": False,
            "showline": True,
            "linecolor": "#e9ecef",
            "linewidth": 2,
            "tickformat": ".1%",
            "automargin": True
        },
        "margin": {"l": 60, "r": 30, "t": 40, "b": 60},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Plus Jakarta Sans", "size": 13, "color": "#495057"},
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12, "color": "#1a1a1a"}
        }
    }
    
    return {"data": data, "layout": layout}

def create_yield_curve_chart(maturities, yields):
    """
    Create the implied yield curve chart.
    """
    data = [{
        "x": maturities.tolist() if isinstance(maturities, np.ndarray) else maturities,
        "y": yields.tolist() if isinstance(yields, np.ndarray) else yields,
        "mode": "lines",
        "name": "Yield Curve",
        "fill": "tozeroy",
        "fillcolor": "rgba(255, 102, 0, 0.1)",
        "line": {"color": "#ff6600", "width": 3}
    }]

    layout = {
        "template": "plotly_white",
        "xaxis": {
            "title": {"text": "Maturity (Years)", "standoff": 20},
            "showgrid": True,
            "gridcolor": "#f0f2f5",
            "showline": True,
            "linecolor": "#e9ecef",
            "linewidth": 2,
            "automargin": True
        },
        "yaxis": {
            "title": {"text": "Yield", "standoff": 20},
            "showgrid": True,
            "gridcolor": "#f0f2f5",
            "showline": True,
            "linecolor": "#e9ecef",
            "linewidth": 2,
            "tickformat": ".1%",
            "automargin": True
        },
        "margin": {"l": 60, "r": 30, "t": 30, "b": 60},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Plus Jakarta Sans", "size": 13, "color": "#495057"},
        "hovermode": "x unified"
    }
    
    return {"data": data, "layout": layout}

def create_distribution_chart(mu, std, x_axis, y_axis):
    """
    Create a probability distribution chart.
    """
    data = [{
        "x": x_axis.tolist() if isinstance(x_axis, np.ndarray) else x_axis,
        "y": y_axis.tolist() if isinstance(y_axis, np.ndarray) else y_axis,
        "mode": "lines",
        "name": "Probability Density",
        "fill": "tozeroy",
        "fillcolor": "rgba(255, 102, 0, 0.1)",
        "line": {"color": "#ff6600", "width": 3}
    }]
    
    layout = {
        "template": "plotly_white",
        "xaxis": {
            "title": {"text": "Interest Rate", "standoff": 20},
            "tickformat": ".1%",
            "showgrid": True,
            "gridcolor": "#f0f2f5",
            "showline": True,
            "linecolor": "#e9ecef",
            "linewidth": 2,
            "automargin": True
        },
        "yaxis": {
            "title": {"text": "Probability Density", "standoff": 20},
            "showgrid": True,
            "gridcolor": "#f0f2f5",
            "showline": True,
            "linecolor": "#e9ecef",
            "linewidth": 2,
            "showticklabels": False,
            "automargin": True
        },
        "margin": {"l": 60, "r": 30, "t": 30, "b": 60},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Plus Jakarta Sans", "size": 13, "color": "#495057"},
        "shapes": [{
            "type": "line",
            "x0": mu, "x1": mu,
            "y0": 0, "y1": 1,
            "yref": "paper",
            "line": {"color": "#1a1a1a", "width": 2, "dash": "dash"}
        }]
    }
    
    return {"data": data, "layout": layout}