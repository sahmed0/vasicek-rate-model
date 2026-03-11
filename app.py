import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from vasicek import simulate_vasicek, calculate_expected_path, calculate_yield_curve, calculate_future_distribution
from charts import create_simulation_chart, create_yield_curve_chart, create_distribution_chart
# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vasicek Rate Model",
    page_icon="hmC",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #F9FAFB;
        color: #111827;
    }
    
    h1, h2, h3 {
        font-weight: 700;
        color: #111827;
        letter-spacing: -1px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #2563EB;
        letter-spacing: -1px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 54px;
        background-color: transparent;
        border-bottom: 2px solid transparent;
        color: #4B5563;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 3px solid #2563EB;
        color: #2563EB;
        font-weight: 700;
    }
    
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        font-weight: 600;
        color: #111827;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stExpander"] details {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        background-color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HEADER & CONFIGURATION ---
col_logo, col_title = st.columns([1, 10])

with col_title:
    st.title(" 📈 Vasicek Rate Model")
    st.markdown("Interactive Interest Rate Modelling by solving the Vasicek SDE")

st.markdown("---")

# --- 4. INPUTS ---
with st.expander("⚙️ Model Configuration (Click to Expand)", expanded=True):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        currentRatePct = st.number_input("Current Rate ($$r_0$$) %", value=3.75, step=0.01, format="%.2f")
        current_rate = currentRatePct / 100.0
    with c2:
        longTermMeanPct = st.number_input("Long Term Mean ($$b$$) %", value=2.65, step=0.01, format="%.2f")
        long_term_mean = longTermMeanPct / 100.0
    with c3:
        reversion_speed = st.number_input("Reversion Speed ($$a$$)", value=0.30, step=0.01, format="%.2f")
    with c4:
        volatility = st.number_input("Volatility ($$σ$$)", value=0.020, step=0.001, format="%.3f")
    with c5:
        time_horizon = st.number_input("Time Horizon ($$Yrs$$)", value=10, step=1)
    with c6:
        num_simulations = st.number_input("Simulations ($$N$$)", value=100, step=10)

# --- 5. MAIN CONTENT TABS ---
tab_sim, tab_yield, tab_dist, tab_maths, tab_code = st.tabs([
    "📈 Interest Rate Model", 
    "📊 Bond Yield Curve", 
    "🎯 Probability Forecast",
    "🧮 The Mathematics",
    "💻 The Code"
])

# ==========================================
# TAB 1: VASICEK SIMULATION
# ==========================================
with tab_sim:
    st.markdown("### Vasicek Interest Rate Model")
    st.caption(f"Visualising {num_simulations} stochastic paths evolving over {time_horizon} years. Each path is calculated using the Euler-Maruyama method to solve the Vasicek SDE.")
    
    time_axis, rate_paths = simulate_vasicek(
        r0=current_rate,
        a=reversion_speed,
        b=long_term_mean,
        sigma=volatility,
        T=time_horizon,
        num_sims=num_simulations
    )

    expected_path = calculate_expected_path(current_rate, reversion_speed, long_term_mean, time_horizon)
    
    fig_sim = create_simulation_chart(time_axis, rate_paths, expected_path, num_simulations, long_term_mean, time_horizon)
    st.plotly_chart(fig_sim, use_container_width=True)

# ==========================================
# TAB 2: YIELD CURVE
# ==========================================
with tab_yield:
    st.markdown("### Term Structure of Interest Rates")
    st.caption("This is the Implied Zero-Coupon Yield Curve based on the current parameters.")
    
    yc_maturities, yc_yields = calculate_yield_curve(current_rate, reversion_speed, long_term_mean, volatility, time_horizon)
    fig_yc = create_yield_curve_chart(yc_maturities, yc_yields)
    st.plotly_chart(fig_yc, use_container_width=True)
    
    if yc_yields[-1] < yc_yields[0]:
        st.warning("⚠️ **Inverted Yield Curve:** Long-term rates are lower than short-term rates (Recession Signal).")
    else:
        st.success("✅ **Normal Yield Curve:** Upward sloping structure indicating economic expansion.")

# ==========================================
# TAB 3: PROBABILITY FORECAST
# ==========================================
with tab_dist:
    st.markdown("### Rate Probability Distribution")
    st.caption("Analytical forecast of interest rate probabilities at a specific future date.")

    col_input, col_stats = st.columns([1, 3])
    with col_input:
        forecast_year = st.slider("Select Horizon (Years)", 0.1, float(time_horizon), 1.0, 0.1)
    
    mu, std = calculate_future_distribution(current_rate, reversion_speed, long_term_mean, volatility, forecast_year)
    upper_95 = mu + 1.96 * std
    lower_95 = mu - 1.96 * std
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Expected Rate", f"{mu:.2%}")
    k2.metric("Uncertainty (Std Dev)", f"±{std:.2%}")
    k3.metric("95% Conf. Interval", f"{lower_95:.2%} - {upper_95:.2%}")

    x_axis = np.linspace(mu - 4*std, mu + 4*std, 1000)
    y_axis = norm.pdf(x_axis, mu, std)
    
    fig_dist = create_distribution_chart(mu, std, x_axis, y_axis)
    st.plotly_chart(fig_dist, use_container_width=True)

# ==========================================
# TAB 4: THE MATHEMATICS
# ==========================================
with tab_maths:
    st.markdown("### 1. The Vasicek Model SDE")
    st.markdown(r"""
    The Vasicek model describes the evolution of interest rates as a **Stochastic Differential Equation (SDE)**.
    It captures the tension between random market shocks and a long-term economic equilibrium.
    
    $$ dr_t = a(b - r_t)dt + \sigma dW_t $$
    
    Where:
    * $r_t$: The instantaneous short rate at time $t$.
    * $a$: **Speed of Reversion**. The gravitational pull back to equilibrium.
    * $b$: **Long-Term Mean**. The equilibrium level (the red line on the chart).
    * $\sigma$: **Volatility**. The magnitude of the random shocks.
    * $dW_t$: **Wiener Process** (Brownian Motion). $dW_t \sim N(0, dt)$.
    """)
    
    st.markdown("---")
    
    st.markdown("### 2. The Ornstein-Uhlenbeck Solution")
    st.markdown(r"""
    This SDE is a specific case of the Ornstein-Uhlenbeck process and can be solved explicitly using Itō's Lemma:
    
    $$ r_t = r_0 e^{-at} + b(1 - e^{-at}) + \sigma \int_0^t e^{-a(t-s)} dW_s $$
    
    * **Deterministic Part:** $r_0 e^{-at} + b(1 - e^{-at})$ (This creates the "Expected Path" dashed line).
    * **Stochastic Part:** $\sigma \int_0^t e^{-a(t-s)} dW_s$ (This creates the "random noise" in the simulation).
    """)

    st.markdown("---")

    st.markdown("### 3. Bond Pricing (Affine Term Structure)")
    st.markdown(r"""
    One of the Vasicek model's strengths is that it produces a closed-form solution for bond prices.
    The price of a Zero-Coupon Bond $P(t,T)$ maturing at time $T$ is:
    
    $$ P(t, T) = A(t, T) e^{-B(t, T) r_t} $$
    
    This relationship implies that the Yield Curve is determined entirely by the short rate $r_t$ and the model parameters.
    """)

# ==========================================
# TAB 5: THE CODE (IMPLEMENTATION)
# ==========================================
with tab_code:
    st.markdown("### 1. Euler-Maruyama Discretisation")
    st.markdown("""
    Since continuous time ($$ dt_0 $$) is impossible to model on a discrete computer, we approximate the Vasicek SDE using the **Euler-Maruyama method**.
    
    This transforms the differential equation into an iterative update rule:
    """)
    
    st.code("""
    # Loop through time steps
    for t in range(1, N):
        # 1. Deterministic Drift (Pull to Mean)
        drift = a * (b - rates[t-1]) * dt
        
        # 2. Stochastic Shock (Random Noise)
        # We scale the random number by sqrt(dt) because variance is linear with time.
        shock = sigma * np.sqrt(dt) * np.random.normal()
        
        # 3. Update Rate
        rates[t] = rates[t-1] + drift + shock
    """, language="python")
    
    st.markdown("### 2. Probability Forecast")
    st.markdown("""
    To generate the "Probability Forecast" tab, we do not need to run simulations. We use the statistical properties of the normal distribution derived from the model:
    """)
    
    st.code("""
    def calculate_future_distribution(r0, a, b, sigma, t):
        # Expected Value (Mean)
        expected_mean = r0 * np.exp(-a * t) + b * (1 - np.exp(-a * t))
        
        # Variance (Derived from Ito Isometry)
        variance = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
        
        return expected_mean, np.sqrt(variance)
    """, language="python")
    