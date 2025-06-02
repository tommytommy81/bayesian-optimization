# Import necessary libraries
import pandas as pd
import numpy as np
import pystan
import matplotlib.pyplot as plt

# Load the investment decisions data
# Assuming df is your subject's DataFrame, and distro_estimates is as defined
FACTOR = 1/100
beta = 0.88

# Prepare data
gain = df['gain'].values * FACTOR
loss = df['loss'].values * FACTOR
prob_win = df['prob_win'].values
prob_loss = df['prob_loss'].values
prob_ambi = df['prob_ambi'].values
invest = df['invest'].values.astype(int)

# Prepare data for Stan
stan_data = {
    'N': len(gain),
    'gain': gain,
    'loss': loss,
    'prob_win': prob_win,
    'prob_loss': prob_loss,
    'prob_ambi': prob_ambi,
    'invest': invest,
    'mu_theta': distro_estimates['mu']['theta'],
    'sigma_theta': distro_estimates['sigma']['theta'],
    'mu_Lambda': distro_estimates['mu']['Lambda'],
    'sigma_Lambda': distro_estimates['sigma']['Lambda'],
    'mu_tau': distro_estimates['mu']['tau'],
    'sigma_tau': distro_estimates['sigma']['tau'],
    'mu_alpha': distro_estimates['mu']['alpha'],
    'sigma_alpha': distro_estimates['sigma']['alpha'],
    'mu_gamma': distro_estimates['mu']['gamma'],
    'sigma_gamma': distro_estimates['sigma']['gamma'],
    'sigma_error': distro_estimates['sigma']['error'],
}

# Define the Stan model
stan_model_code = """
data {
    int<lower=0> N; // number of observations
    vector[N] gain; // gains
    vector[N] loss; // losses
    vector[N] prob_win; // probability of winning
    vector[N] prob_loss; // probability of losing
    vector[N] prob_ambi; // probability of ambiguity
    int<lower=0, upper=1> invest[N]; // observed investment decisions

    real mu_theta;
    real<lower=0> sigma_theta;
    real mu_Lambda;
    real<lower=0> sigma_Lambda;
    real mu_tau;
    real<lower=0> sigma_tau;
    real mu_alpha;
    real<lower=0> sigma_alpha;
    real mu_gamma;
    real<lower=0> sigma_gamma;
    real<lower=0> sigma_error;
}

parameters {
    real<lower=0> theta; // parameter theta
    real<lower=0> Lambda; // parameter Lambda
    real<lower=0> tau; // parameter tau
    real<lower=0> alpha; // parameter alpha
    real<lower=0> gamma; // parameter gamma
    real error; // error term
}

model {
    // Priors
    theta ~ lognormal(mu_theta, sigma_theta);
    Lambda ~ lognormal(mu_Lambda, sigma_Lambda);
    tau ~ lognormal(mu_tau, sigma_tau);
    alpha ~ lognormal(mu_alpha, sigma_alpha);
    gamma ~ lognormal(mu_gamma, sigma_gamma);
    error ~ normal(0, sigma_error);

    // Utility and probability
    for (n in 1:N) {
        real prob_g = prob_win[n] + prob_ambi[n] * theta;
        real prob_l = prob_loss[n] + prob_ambi[n] * (1 - theta);
        real utility = (gain[n] >= 0 ? pow(gain[n], alpha) : -Lambda * pow(-gain[n], beta)) * pow(prob_g, gamma) +
                       (loss[n] >= 0 ? pow(loss[n], alpha) : -Lambda * pow(-loss[n], beta)) * pow(prob_l, gamma);
        invest[n] ~ bernoulli_logit(tau * (utility - error));
    }
}
"""

# Compile the model
stan_model = pystan.StanModel(model_code=stan_model_code)

# Fit the model
fit = stan_model.sampling(data=stan_data, iter=1000, chains=4, control={'adapt_delta': 0.9})

# Print the results
print(fit)

# Extract the samples
trace = fit.extract()

# Plot the results
plt.figure(figsize=(10, 6))
plt.hist(trace['theta'], bins=30, alpha=0.5, label='Theta')
plt.hist(trace['Lambda'], bins=30, alpha=0.5, label='Lambda')
plt.hist(trace['tau'], bins=30, alpha=0.5, label='Tau')
plt.hist(trace['alpha'], bins=30, alpha=0.5, label='Alpha')
plt.hist(trace['gamma'], bins=30, alpha=0.5, label='Gamma')
plt.legend()
plt.title('Posterior Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()