import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def monte_carlo_bs(S, K, T, r, sigma, option_type='call', n_sims=10000, seed=42):
    np.random.seed(seed)
    Z = np.random.standard_normal(n_sims)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0) if option_type == 'call' else np.maximum(K - S_T, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    se = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    return price, se

def monte_carlo_bs_antithetic(S, K, T, r, sigma, option_type='call', n_sims=10000, seed=42):
    np.random.seed(seed)
    n_half = n_sims // 2
    Z = np.random.standard_normal(n_half)
    Z_full = np.concatenate([Z, -Z])
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_full)
    payoffs = np.maximum(S_T - K, 0) if option_type == 'call' else np.maximum(K - S_T, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    se = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    return price, se

# Cache the expensive convergence sweep so it doesn't rerun on every slider tick
@st.cache_data
def run_convergence_sweep(S, K, T, r, sigma, option_type):
    sim_counts = np.logspace(2, 5, 25).astype(int)
    std_results = [monte_carlo_bs(S, K, T, r, sigma, option_type, n_sims=n) for n in sim_counts]
    anti_results = [monte_carlo_bs_antithetic(S, K, T, r, sigma, option_type, n_sims=n) for n in sim_counts]
    return sim_counts, std_results, anti_results

# ---------- Website layout ----------

st.set_page_config(page_title="Black-Scholes & Monte Carlo Pricer", layout="wide")
st.title("Option Pricer: Black-Scholes vs. Monte Carlo")

# Sliders live in the sidebar so they persist across tabs
S = st.sidebar.number_input("Spot Price (S)", min_value=0.01, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (years)", min_value=0.01, max_value=10.0, value=1.0, step=0.05)
r = st.sidebar.number_input("Risk-free Rate (r)", min_value=0.0, max_value=0.20, value=0.05, step=0.005, format="%.3f")
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.20, step=0.01, format="%.3f")
option_type = st.sidebar.radio("Option Type", ['call', 'put'])

tab1, tab2 = st.tabs(["Pricing Calculator", "Monte Carlo Analysis"])

# ---------- Tab 1: fast, live-updating ----------
with tab1:
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    mc_price, mc_se = monte_carlo_bs_antithetic(S, K, T, r, sigma, option_type, n_sims=20000)

    col1, col2 = st.columns(2)
    col1.metric("Black-Scholes Price", f"{bs_price:.4f}")
    col2.metric("Monte Carlo Price (antithetic)", f"{mc_price:.4f}", f"±{1.96*mc_se:.4f} (95% CI)")

    st.subheader("Sensitivity: Price vs. Spot Price")
    S_range = np.linspace(max(0.01, S * 0.5), S * 1.5, 100)
    prices = [black_scholes(s, K, T, r, sigma, option_type) for s in S_range]
    fig, ax = plt.subplots()
    ax.plot(S_range, prices)
    ax.axvline(S, color='gray', linestyle='--', label='Current S')
    ax.set_xlabel("Spot Price"); ax.set_ylabel("Option Price"); ax.legend()
    st.pyplot(fig)

# ---------- Tab 2: heavier, cached, explicit run ----------
with tab2:
    st.write("This runs a full convergence sweep (100 to 100,000 simulations) using your current sidebar parameters.")
    if st.button("Run Monte Carlo Analysis"):
        with st.spinner("Running simulations..."):
            sim_counts, std_results, anti_results = run_convergence_sweep(S, K, T, r, sigma, option_type)

        std_prices = [x[0] for x in std_results]
        anti_prices = [x[0] for x in anti_results]
        std_ci = [1.96 * x[1] for x in std_results]
        anti_ci = [1.96 * x[1] for x in anti_results]
        bs_ref = black_scholes(S, K, T, r, sigma, option_type)

        st.subheader("Convergence: Standard vs. Antithetic MC")
        fig1, ax1 = plt.subplots()
        ax1.plot(sim_counts, std_prices, marker='o', label='Standard MC', alpha=0.7)
        ax1.plot(sim_counts, anti_prices, marker='s', label='Antithetic MC', alpha=0.7)
        ax1.axhline(bs_ref, color='red', linestyle='--', label='Black-Scholes')
        ax1.set_xscale('log'); ax1.set_xlabel("Number of Simulations"); ax1.set_ylabel("Price"); ax1.legend()
        st.pyplot(fig1)

        st.subheader("Variance Reduction: 95% CI Width")
        fig2, ax2 = plt.subplots()
        ax2.plot(sim_counts, std_ci, marker='o', label='Standard MC')
        ax2.plot(sim_counts, anti_ci, marker='s', label='Antithetic MC')
        ax2.set_xscale('log'); ax2.set_yscale('log')
        ax2.set_xlabel("Number of Simulations"); ax2.set_ylabel("95% CI Width"); ax2.legend()
        st.pyplot(fig2)