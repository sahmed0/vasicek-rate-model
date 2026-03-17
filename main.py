import js
import asyncio

# --- Console Logs for Debugging ---
def log_to_console(message, type="info"):
    prefix = "[VASICEK-DEBUG]"
    msg_str = str(message)
    if type == "info":
        js.console.log(f"{prefix} {msg_str}")
    elif type == "error":
        js.console.error(f"{prefix} ❌ {msg_str}")
    elif type == "success":
        js.console.log(f"{prefix} ✅ {msg_str}")

async def init_app():
    log_to_console("Beginning module audit...")
    try:
        from pyscript import document, window
        from pyodide.ffi import create_proxy, to_js
        log_to_console("PyScript core utilities found.")
    except Exception as e:
        log_to_console(f"PyScript internal error: {e}", type="error")
        return

    # audit imports one by one
    try:
        log_to_console("Auditing numpy...")
        import numpy as np
        log_to_console("numpy ready.", type="success")
        
        log_to_console("Auditing scipy...")
        from scipy.stats import norm
        log_to_console("scipy.stats ready.", type="success")
        
        log_to_console("Auditing internal modules...")
        from vasicek import simulate_vasicek, calculate_expected_path, calculate_yield_curve, calculate_future_distribution
        from charts import create_simulation_chart, create_yield_curve_chart, create_distribution_chart
        log_to_console("Internal modules ready.", type="success")
        
    except Exception as e:
        log_to_console(f"Dependency failure: {e}", type="error")
        err_msg = f"Error: {e}. Check console for details."
        subtext = document.getElementById("loading-subtext")
        if subtext:
            subtext.innerText = err_msg
        return

    # --- App Setup ---
    def render_charts(event=None):
        try:
            r0 = float(document.getElementById("current_rate").value) / 100.0
            b = float(document.getElementById("long_term_mean").value) / 100.0
            a = float(document.getElementById("reversion_speed").value)
            sigma = float(document.getElementById("volatility").value)
            T = int(document.getElementById("time_horizon").value)
            
            # Handle potential duplicate num_simulations IDs (prioritise number input)
            num_sim_els = document.querySelectorAll("#num_simulations")
            num_sims = 100
            for el in num_sim_els:
                if el.tagName == "INPUT" and el.type != "hidden":
                    num_sims = int(el.value)
                    break
            
            forecast_horizon = float(document.getElementById("forecast_slider").value)

            # 1. Vasicek Simulation
            time_axis, rate_paths = simulate_vasicek(r0, a, b, sigma, T=T, num_sims=num_sims)
            expected_path = calculate_expected_path(r0, a, b, T=T)
            config_sim = create_simulation_chart(time_axis, rate_paths, expected_path, num_sims, b, T)
            
            js.Plotly.newPlot(
                "chart-sim", 
                to_js(config_sim['data'], dict_converter=js.Object.fromEntries), 
                to_js(config_sim['layout'], dict_converter=js.Object.fromEntries), 
                to_js({"responsive": True, "displayModeBar": False}, dict_converter=js.Object.fromEntries)
            )

            # 2. Yield Curve
            yc_maturities, yc_yields = calculate_yield_curve(r0, a, b, sigma, max_maturity=T)
            config_yc = create_yield_curve_chart(yc_maturities, yc_yields)
            js.Plotly.newPlot(
                "chart-yield", 
                to_js(config_yc['data'], dict_converter=js.Object.fromEntries), 
                to_js(config_yc['layout'], dict_converter=js.Object.fromEntries), 
                to_js({"responsive": True, "displayModeBar": False}, dict_converter=js.Object.fromEntries)
            )

            # 3. Probability Distribution
            mu, std = calculate_future_distribution(r0, a, b, sigma, forecast_horizon)
            document.getElementById("metric-expected").innerText = f"{mu:.2%}"
            document.getElementById("metric-std").innerText = f"±{std:.2%}"
            document.getElementById("metric-conf").innerText = f"{(mu-1.96*std):.2%} - {(mu+1.96*std):.2%}"

            x_axis = np.linspace(mu - 4*std, mu + 4*std, 1000)
            y_axis = norm.pdf(x_axis, mu, std)
            config_dist = create_distribution_chart(mu, std, x_axis, y_axis)
            js.Plotly.newPlot(
                "chart-dist", 
                to_js(config_dist['data'], dict_converter=js.Object.fromEntries), 
                to_js(config_dist['layout'], dict_converter=js.Object.fromEntries), 
                to_js({"responsive": True, "displayModeBar": False}, dict_converter=js.Object.fromEntries)
            )

        except Exception as e:
            log_to_console(f"Render Error: {e}", type="error")

    # Setup listeners
    proxy = create_proxy(render_charts)
    for inp_id in ["current_rate", "long_term_mean", "reversion_speed", "volatility", "time_horizon", "num_simulations", "forecast_slider"]:
        document.getElementById(inp_id).addEventListener("input", proxy)

    # Resize listener for fluid grid layouts
    def on_resize(event):
        js.window.dispatchEvent(js.Event.new("resize"))
    
    window.addEventListener("resize", create_proxy(on_resize))

    # Initial render
    render_charts()
    window.hideLoader()
    log_to_console("Application ready.", type="success")

# Run the async init
asyncio.ensure_future(init_app())
