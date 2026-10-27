
import numpy as np 
from scipy.stats import norm

#black-scholes formula to price european call and put options
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

#check the function with example parameters
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

call_price = black_scholes(S, K, T, r, sigma, 'call')
put_price = black_scholes(S, K, T, r, sigma, 'put')

print(f"Call price: {call_price:.4f}")  # should be ~10.45
print(f"Put price: {put_price:.4f}")    # should be ~5.57

#plotting the option prices wrt the diff parameters
import matplotlib.pyplot as plt
# Base case
S0, K0, T0, r0, sigma0 = 100, 100, 1, 0.05, 0.2

# 1. Price vs Spot
S_range = np.linspace(50, 150, 100)
call_prices = [black_scholes(S, K0, T0, r0, sigma0, 'call') for S in S_range]
put_prices = [black_scholes(S, K0, T0, r0, sigma0, 'put') for S in S_range]

plt.figure(figsize=(8, 5))
plt.plot(S_range, call_prices, label='Call')
plt.plot(S_range, put_prices, label='Put')
plt.axvline(K0, color='gray', linestyle='--', label='Strike (K)')
plt.xlabel('Spot Price (S)')
plt.ylabel('Option Price')
plt.title('Option Price vs. Spot Price')
plt.legend()
plt.show()

# 2. Price vs Volatility
sigma_range = np.linspace(0.05, 0.6, 100)
call_vs_sigma = [black_scholes(S0, K0, T0, r0, sig, 'call') for sig in sigma_range]
put_vs_sigma = [black_scholes(S0, K0, T0, r0, sig, 'put') for sig in sigma_range]

plt.figure(figsize=(8, 5))
plt.plot(sigma_range, call_vs_sigma, label='Call')
plt.plot(sigma_range, put_vs_sigma, label='Put')
plt.xlabel('Volatility (σ)')
plt.ylabel('Option Price')
plt.title('Option Price vs. Volatility')
plt.legend()
plt.show()

# 3. Price vs Time to Maturity
T_range = np.linspace(0.01, 2, 100)
call_vs_T = [black_scholes(S0, K0, T, r0, sigma0, 'call') for T in T_range]
put_vs_T = [black_scholes(S0, K0, T, r0, sigma0, 'put') for T in T_range]

plt.figure(figsize=(8, 5))
plt.plot(T_range, call_vs_T, label='Call')
plt.plot(T_range, put_vs_T, label='Put')
plt.xlabel('Time to Maturity (years)')
plt.ylabel('Option Price')
plt.title('Option Price vs. Time to Maturity')
plt.legend()
plt.show()


#checking for convergence using Monte Carlo simulation (GBM) 
def monte_carlo_bs(S, K, T, r, sigma, option_type='call', n_sims=10000, seed=42):
    np.random.seed(seed)  # reproducibility
    Z = np.random.standard_normal(n_sims)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)  # for confidence interval later
    return price, std_error

#check against BS model
mc_call, se = monte_carlo_bs(S0, K0, T0, r0, sigma0, 'call', n_sims=100000)
bs_call = black_scholes(S0, K0, T0, r0, sigma0, 'call')
print(f"MC price: {mc_call:.4f} (±{1.96*se:.4f} at 95% CI)")
print(f"BS price: {bs_call:.4f}")

def monte_carlo_bs_antithetic(S, K, T, r, sigma, option_type='call', n_sims=10000, seed=42):
    np.random.seed(seed)
    n_half = n_sims // 2
    Z = np.random.standard_normal(n_half)
    Z_antithetic = np.concatenate([Z, -Z])  # pairs every Z with its mirror -Z

    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_antithetic)

    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    return price, std_error

sim_counts = np.logspace(2, 5, 30).astype(int)

# Standard MC: collect price and std error at each sim count
standard_results = [monte_carlo_bs(S0, K0, T0, r0, sigma0, 'call', n_sims=n) for n in sim_counts]
standard_prices = [r[0] for r in standard_results]
standard_ci_width = [1.96 * r[1] for r in standard_results]

# Antithetic MC: same thing
antithetic_results = [monte_carlo_bs_antithetic(S0, K0, T0, r0, sigma0, 'call', n_sims=n) for n in sim_counts]
antithetic_prices = [r[0] for r in antithetic_results]
antithetic_ci_width = [1.96 * r[1] for r in antithetic_results]

# Plot 1: both converge to the same BS price
plt.figure(figsize=(8, 5))
plt.plot(sim_counts, standard_prices, marker='o', label='Standard MC', alpha=0.7)
plt.plot(sim_counts, antithetic_prices, marker='s', label='Antithetic MC', alpha=0.7)
plt.axhline(bs_call, color='red', linestyle='--', label='Black-Scholes price')
plt.xscale('log')
plt.xlabel('Number of Simulations')
plt.ylabel('Call Option Price')
plt.title('Standard vs. Antithetic MC — Convergence')
plt.legend()
plt.show()

# Plot 2: the actual payoff — antithetic has a tighter CI at every N
plt.figure(figsize=(8, 5))
plt.plot(sim_counts, standard_ci_width, marker='o', label='Standard MC 95% CI width')
plt.plot(sim_counts, antithetic_ci_width, marker='s', label='Antithetic MC 95% CI width')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Simulations')
plt.ylabel('95% CI Width')
plt.title('Variance Reduction: CI Width Comparison')
plt.legend()
plt.show()