import pymc as pm
import pandas as pd

# Load the investment decisions data
data = pd.read_csv('../data/investment_decisions.csv')

# Define the PyMC model
with pm.Model() as model:
    # Informative priors for parameters
    alpha = pm.Normal('alpha', mu=0, sigma=1)  # Prior for alpha
    beta_theta = pm.Normal('beta_theta', mu=0, sigma=1)  # Prior for theta
    beta_lambda = pm.Normal('beta_lambda', mu=0, sigma=1)  # Prior for lambda
    beta_alpha = pm.Normal('beta_alpha', mu=0, sigma=1)  # Prior for alpha
    beta_gamma = pm.Normal('beta_gamma', mu=0, sigma=1)  # Prior for gamma

    # Linear combination of predictors
    linear_combination = (beta_theta * data['prob_win'] +
                          beta_lambda * data['gain'] +
                          beta_alpha * data['loss'] +
                          beta_gamma * data['error'] +
                          alpha)

    # Bernoulli likelihood for investment decisions
    investment_outcome = pm.Bernoulli('investment_outcome', logit_p=linear_combination, observed=data['investment'])

    # Posterior sampling
    trace = pm.sample(2000, tune=1000, return_inferencedata=False)  # Adjust the number of samples and tuning as needed

# The model is now defined and can be used for further analysis and diagnostics.