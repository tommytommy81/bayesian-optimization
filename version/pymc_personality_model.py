# This script:

# Defines utility and decision logic using your Prospect Theory model

# Sets priors using log-normal and normal distributions

# Builds a PyMC model using observed data

# Performs MCMC sampling via pm.sample()

# You just need to:

# Load your DataFrame (data) with columns:
# ['invest', 'potential_gain', 'potential_loss', 'probability_win_percent', 'probability_loss_percent']

# Call build_pymc_model(data)




import pymc as pm
import numpy as np
import pandas as pd
import aesara.tensor as at

def calc_eta(green, red, theta):
    return at.clip(green + (1 - green - red) * theta, 0, 1)

def calc_subj_prob(p, gamma):
    return (p ** gamma) / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))

def calc_subj_values(x, Lambda, alpha, beta):
    return at.switch(x >= 0, x ** alpha, -Lambda * ((-x) ** beta))

def pt_utility(gain, loss, p_win, p_loss, theta, Lambda, alpha, beta, gamma):
    eta_gain = calc_eta(p_win, p_loss, theta)
    eta_loss = calc_eta(p_loss, p_win, 1 - theta)
    v_gain = calc_subj_values(gain, Lambda, alpha, beta)
    v_loss = calc_subj_values(-loss, Lambda, alpha, beta)
    pi_gain = calc_subj_prob(eta_gain, gamma)
    pi_loss = calc_subj_prob(eta_loss, gamma)
    return v_gain * pi_gain + v_loss * pi_loss

def build_pymc_model(data):
    with pm.Model() as model:
        # Priors based on your lognormal + normal distributions
        theta = pm.Lognormal('theta', mu=0, sigma=1)
        Lambda = pm.Lognormal('Lambda', mu=0, sigma=1)
        tau = pm.Lognormal('tau', mu=0, sigma=1)
        alpha = pm.Lognormal('alpha', mu=0, sigma=1)
        beta = pm.Lognormal('beta', mu=0, sigma=1)
        gamma = pm.Lognormal('gamma', mu=0, sigma=1)
        error = pm.Normal('error', mu=0, sigma=1)

        # Observed data
        gain = pm.Data('gain', data['potential_gain'])
        loss = pm.Data('loss', data['potential_loss'])
        p_win = pm.Data('p_win', data['probability_win_percent'])
        p_loss = pm.Data('p_loss', data['probability_loss_percent'])
        invest = pm.Data('invest', data['invest'])

        # Calculate utility and decision probability
        utility = pt_utility(gain, loss, p_win, p_loss, theta, Lambda, alpha, beta, gamma)
        prob = pm.Deterministic('prob', 1 / (1 + at.exp(-tau * (utility - error))))

        # Likelihood
        y_obs = pm.Bernoulli('y_obs', p=prob, observed=invest)

        # Sampling
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    return model, trace
