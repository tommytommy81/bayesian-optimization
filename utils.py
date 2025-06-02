
# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az
import stan
import aesara.tensor as at
import nest_asyncio

from scipy.special import ndtri
from scipy.stats import norm, lognorm
from scipy.optimize import minimize


from sklearn.metrics import roc_auc_score

import time



def stan_wrapper(df, x0, FACTOR=1, beta = 0.88):
    """
    Wrapper function to prepare data for Stan model.
    """
    
    
    # Ensure the DataFrame has the required columns
    required_columns = ['gain', 'loss', 'prob_win', 'prob_loss', 'prob_ambi', 'invest']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column: {col}")



    # Convert 'invest' to binary (0 or 1)
    df['invest'] = df['invest'].apply(lambda x: 1 if x > 0 else 0)

    # # Add beta to stan_data, as it is used in the Stan model
    # beta = 0.88
    
    mu_theta, sig_theta, mu_Lambda, sig_Lambda, mu_tau, sig_tau, mu_alpha, sig_alpha, mu_gamma, sig_gamma, sig_error = x0

    # Prepare stan_data dictionary
    stan_data = {
        'N': len(df),
        'gain': df['gain'].values* FACTOR,
        'loss': df['loss'].values* FACTOR,
        'prob_win': df['prob_win'].values,
        'prob_loss': df['prob_loss'].values,
        'prob_ambi': df['prob_ambi'].values,
        'invest': df['invest'].astype(int).values,
        'mu_theta': mu_theta,
        'sigma_theta': sig_theta,
        'mu_Lambda': mu_Lambda,
        'sigma_Lambda': sig_Lambda,
        'mu_tau': mu_tau,
        'sigma_tau': sig_tau,
        'mu_alpha': mu_alpha,
        'sigma_alpha': sig_alpha,
        'mu_gamma': mu_gamma,
        'sigma_gamma': sig_gamma,
        'sigma_error': sig_error,
        'beta': beta
    }

    nest_asyncio.apply()


    # # Set initial values for the parameters
    # init_values = {
    #     'theta': 0,
    #     'Lambda': LAMBDA_INIT,
    #     'tau': TAU_INIT,
    #     'alpha': ALPHA_INIT,
    #     'gamma': GAMMA_INIT,
    #     'error': ERROR_INIT
    # }



    # Update Stan model code to accept beta as data
    stan_model_code = """
    data {
        int<lower=0> N; // number of observations
        vector[N] gain; // gains
        vector[N] loss; // losses
        vector[N] prob_win; // probability of winning
        vector[N] prob_loss; // probability of losing
        vector[N] prob_ambi; // probability of ambiguity
        array[N] int<lower=0, upper=1> invest; // observed investment decisions

        real mu_theta;
        real sigma_theta;
        real mu_Lambda;
        real sigma_Lambda;
        real mu_tau;
        real sigma_tau;
        real mu_alpha;
        real sigma_alpha;
        real mu_gamma;
        real sigma_gamma;
        real sigma_error;
        real beta; // <-- added beta as data
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

        // Likelihood
        for (n in 1:N) {
            real utility = (gain[n] >= 0 ? pow(gain[n], alpha) : -Lambda * pow(-gain[n], beta)) * 
                        pow(prob_win[n] + prob_ambi[n] * theta, gamma) +
                        (loss[n] >= 0 ? pow(loss[n], alpha) : -Lambda * pow(-loss[n], beta)) * 
                        pow(prob_loss[n] + prob_ambi[n] * (1 - theta), gamma);
            
            invest[n] ~ bernoulli_logit(tau * (utility - error));
        }
    }
    """


    # Compile the Stan model (synchronously in Jupyter)
    posterior = stan.build(stan_model_code, data=stan_data)

    # Fit the model
    stan_fit = posterior.sample(num_chains=4, num_samples=1000)
    
    return stan_fit


def stan_visualize_out(stan_fit, figure_flag=True):
	"""
	Function to visualize the output of the Stan model.
	"""
	# Convert the fit to a DataFrame for easier plotting
	samples = stan_fit.to_frame()

	if figure_flag:
		# Plot posterior distributions for all parameters
		fig, axes = plt.subplots(len(samples.columns), 1, figsize=(8, 2 * len(samples.columns)), constrained_layout=True)
		if len(samples.columns) == 1:
			axes = [axes]
		for ax, col in zip(axes, samples.columns):
			ax.hist(samples[col], bins=30, density=True, alpha=0.7)
			ax.set_title(f'Posterior of {col}')
		plt.show()

	print(stan_fit.to_frame().describe()[['theta', 'Lambda', 'tau', 'alpha', 'gamma', 'error']].T)


def stan_predict(stan_fit, df, FACTOR=1):
    """
    Function to predict using the Stan model.
    """
    # Extract the mean of the posterior samples for each parameter
    theta = stan_fit.to_frame()['theta'].mean()
    Lambda = stan_fit.to_frame()['Lambda'].mean()
    alpha = stan_fit.to_frame()['alpha'].mean()
    tau = stan_fit.to_frame()['tau'].mean()
    beta = 0.88  # already defined as 0.88
    gamma = stan_fit.to_frame()['gamma'].mean()
    error = stan_fit.to_frame()['error'].mean()

    predictions = []

    for _, row in df.iterrows():
        gain = row['gain'] * FACTOR
        loss = row['loss'] * FACTOR 
        prob_win = row['prob_win']
        prob_loss = row['prob_loss']
        prob_ambi = row['prob_ambi']

        # Calculate subjective probabilities and values using numpy
        def np_calc_subj_values(x, Lambda, alpha, beta):
            return x**alpha if x >= 0 else -Lambda * ((-x)**beta)

        def np_calc_eta(green, red, theta, prob_ambi):
            return green + prob_ambi * theta

        def np_calc_subj_prob(prob, gamma):
            return prob**gamma / (prob**gamma + (1 - prob)**gamma)**(1/gamma)

        prob_g = np.clip(np_calc_eta(prob_win, prob_loss, theta, prob_ambi), 0, 1)
        prob_l = np.clip(np_calc_eta(prob_loss, prob_win, 1-theta, prob_ambi), 0, 1)
        subj_gain = np_calc_subj_values(gain, Lambda, alpha, beta)
        subj_loss = np_calc_subj_values(loss, Lambda, alpha, beta)
        subj_prob_g = np_calc_subj_prob(prob_g, gamma)
        subj_prob_l = np_calc_subj_prob(prob_l, gamma)
        utility = subj_gain * subj_prob_g + subj_loss * subj_prob_l

        # Probability to invest
        p_invest = 1 / (1 + np.exp(-tau * (utility - error)))
        
        # Predict invest = 1 if p_invest > 0.5, else 0
        invest_pred = int(p_invest > 0.5)
        predictions.append(invest_pred)

    return predictions



# Helper functions rewritten for Aesara tensors

def pymc_wrapper(df, distro_estimates, INIT_VALUES, FACTOR=1, beta=0.88, energy_plot_flag=True, distri_plot_flag=True):
    """
    Wrapper function for PyMC model fitting.
    """
    
    
    gain      = df['gain'].values * FACTOR
    loss      = df['loss'].values * FACTOR
    prob_win  = df['prob_win'].values
    prob_loss = df['prob_loss'].values
    prob_ambi = df['prob_ambi'].values
    invest    = df['invest'].values.astype(int)

    THETA_INIT, LAMBDA_INIT, TAU_INIT, ALPHA_INIT, GAMMA_INIT, ERROR_INIT = INIT_VALUES
    initvals = {
        'theta': THETA_INIT,
        'Lambda': LAMBDA_INIT,
        'tau': TAU_INIT,
        'alpha': ALPHA_INIT,
        'gamma': GAMMA_INIT,
        'error': ERROR_INIT
    }

    def lognormal_pdf(value, mu, sigma):
        ''' Function to calculate the lognormal density'''
        shape  = sigma
        loc    = 0
        scale  = np.exp(mu)
        return lognorm.pdf(value, shape, loc, scale)

    def normal_pdf(value, mu, sigma):
        ''' Function to calculate the lognormal density'''
        loc    = 0
        scale  = sigma
        return norm.pdf(value, loc, scale)

    def calc_eta(green, red, theta, prob_ambi):
        # prob_ambi is always provided in your data
        return green + prob_ambi * theta

    def calc_subj_prob(prob, gamma):
        return prob**gamma / (prob**gamma + (1 - prob)**gamma)**(1/gamma)

    def calc_subj_values(x, Lambda, alpha, beta):
        return at.switch(x >= 0, x**alpha, -Lambda * (-x)**beta)

    def calc_pt_utility(gain, loss, prob_win, prob_loss, theta, Lambda, alpha, beta, gamma, prob_ambi):
        prob_g = at.clip(calc_eta(prob_win, prob_loss, theta, prob_ambi), 0, 1)
        prob_l = at.clip(calc_eta(prob_loss, prob_win, 1-theta, prob_ambi), 0, 1)
        u = (
            calc_subj_values(gain, Lambda, alpha, beta) * calc_subj_prob(prob_g, gamma) +
            calc_subj_values(loss, Lambda, alpha, beta) * calc_subj_prob(prob_l, gamma)
        )
        return u

    def calc_prob_invest(utility, tau, error):
        return 1 / (1 + at.exp(-tau * (utility - error)))
    
    
    with pm.Model() as model:
        # Priors
        theta  = pm.Lognormal('theta', mu=distro_estimates['mu']['theta'], sigma=distro_estimates['sigma']['theta'])
        Lambda = pm.Lognormal('Lambda', mu=distro_estimates['mu']['Lambda'], sigma=distro_estimates['sigma']['Lambda'])
        tau    = pm.Lognormal('tau', mu=distro_estimates['mu']['tau'], sigma=distro_estimates['sigma']['tau'])
        alpha  = pm.Lognormal('alpha', mu=distro_estimates['mu']['alpha'], sigma=distro_estimates['sigma']['alpha'])
        gamma  = pm.Lognormal('gamma', mu=distro_estimates['mu']['gamma'], sigma=distro_estimates['sigma']['gamma'])
        error  = pm.Normal('error', mu=0, sigma=distro_estimates['sigma']['error'])

        # Utility and probability
        utility = calc_pt_utility(
            gain=gain,
            loss=loss,
            prob_win=prob_win,
            prob_loss=prob_loss,
            theta=theta,
            Lambda=Lambda,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            prob_ambi=prob_ambi
        )
        p = calc_prob_invest(utility, tau, error)

        # Likelihood
        invest_obs = pm.Bernoulli('invest_obs', p=p, observed=invest)

        # Sample from the posterior
        trace = pm.sample(5000, tune=5000, target_accept=0.9, chains=4, initvals=initvals, return_inferencedata=True)

        if energy_plot_flag:
            az.plot_energy(trace)
            plt.xlabel("Energy")
            energy = trace.sample_stats["energy"].values.flatten()
            plt.xticks(np.round(np.linspace(-10, 10, num=3), 2))
            plt.show()
            print("X axis values (energy):")
            print(energy)    
            
        if distri_plot_flag:
            az.plot_trace(trace, combined=True)
            plt.show()
            
        # Extract posterior means for each parameter from the trace
        summary = az.summary(trace, var_names=["theta", "Lambda", "tau", "alpha", "gamma", "error"])
        print(summary)
    
    return trace


def pymc_predict(trace, df, FACTOR=1, beta=0.88):
    """
    Function to predict using the PyMC model trace.
    """
    # Extract the mean of the posterior samples for each parameter
    theta   = trace.posterior['theta'].mean().item()
    Lambda  = trace.posterior['Lambda'].mean().item()
    alpha   = trace.posterior['alpha'].mean().item()
    tau     = trace.posterior['tau'].mean().item()
    gamma   = trace.posterior['gamma'].mean().item()
    error   = trace.posterior['error'].mean().item()

    predictions = []

    for _, row in df.iterrows():
        gain = row['gain'] * FACTOR
        loss = row['loss'] * FACTOR 
        prob_win = row['prob_win']
        prob_loss = row['prob_loss']
        prob_ambi = row['prob_ambi']

        def np_calc_subj_values(x, Lambda, alpha, beta):
            return x**alpha if x >= 0 else -Lambda * ((-x)**beta)

        def np_calc_eta(green, red, theta, prob_ambi):
            return green + prob_ambi * theta

        def np_calc_subj_prob(prob, gamma):
            return prob**gamma / (prob**gamma + (1 - prob)**gamma)**(1/gamma)

        prob_g = np.clip(np_calc_eta(prob_win, prob_loss, theta, prob_ambi), 0, 1)
        prob_l = np.clip(np_calc_eta(prob_loss, prob_win, 1-theta, prob_ambi), 0, 1)
        subj_gain = np_calc_subj_values(gain, Lambda, alpha, beta)
        subj_loss = np_calc_subj_values(loss, Lambda, alpha, beta)
        subj_prob_g = np_calc_subj_prob(prob_g, gamma)
        subj_prob_l = np_calc_subj_prob(prob_l, gamma)
        utility = subj_gain * subj_prob_g + subj_loss * subj_prob_l

        # Probability to invest
        p_invest = 1 / (1 + np.exp(-tau * (utility - error)))
        
        # Predict invest = 1 if p_invest > 0.5, else 0
        invest_pred = int(p_invest > 0.5)
        predictions.append(invest_pred)

    return predictions




def MLS_wrapper(df, x0, distro_estimates):
    
    # Calculate eta (perceived probability of a successful investment due ambiguity).
    def calc_eta(green, red, theta, prob_ambi=None):
        ''' Calculates the eta given prob_win, prob_loss, and ambiguity. theta is estimated.'''
        if prob_ambi==None:
            return round(green + (1-green-red)*theta,3)
        else:
            # if prob_ambi+red+green!=1:
            #     print(f'Probabilities do not add up to 0: {prob_ambi+red+green}')
            return round(green + prob_ambi*theta,3)

    # Calculate subject probability (using the perceived eta).
    def calc_subj_prob(prob, gamma=1):
        ''' Converts the preceived probabilities to subjective probability. gamma is estimated.'''
        return prob**gamma/(prob**gamma + (1-prob)**gamma)**(1/gamma)

    # Calculating the subjective value of losses and gains. Lambda estimated
    def calc_subj_values(x, Lambda, alpha, beta):
        ''' Calculates the subjective value of gains and losses, given an alpha parameter. Lambda estimate. Alpha could be estimated but used as constant=0.9.'''
        if x >= 0:
            return x**alpha
        else:
            return -Lambda*(-x)**beta

    # Calculating prospect utility.
    def calc_pt_utility(gain, loss, prob_win, prob_loss, theta, Lambda, alpha, beta, gamma=1, prob_ambi=None):
        ''' Calculates the prospect theory utility of investing. Losses are negative numbers and we just add all the outcomes*probabilities.'''
        # print(max(0,calc_eta(prob_win, prob_loss, theta, prob_ambi)))
        # print(min(max(0,calc_eta(prob_win, prob_loss, theta, prob_ambi)),1))
        prob_g = calc_eta(prob_win, prob_loss, theta, prob_ambi).clip(0,1)
        prob_l = calc_eta(prob_loss, prob_win, 1-theta, prob_ambi).clip(0,1)
        pt_u   = calc_subj_values(gain, Lambda, alpha, beta) * calc_subj_prob(prob_g, gamma=gamma) + calc_subj_values(loss, Lambda, alpha, beta) * calc_subj_prob(prob_l, gamma=gamma)
        return pt_u

    # Calculating the probability to invest.
    def calc_prob_invest_data(invest,utility,tau,error):
        yy = 2*invest-1
        return (1)/(1+np.exp(-tau*(utility-error)*yy))

    # classify whether somebody invests or not.
    def likelihood_data(invest,gain, loss, prob_win, prob_loss, theta, Lambda, tau, alpha, beta, gamma,error):
        ''' Function to determine if someone would invest in the gamble given theta, gamma, lambda, and tau.'''
        utility = calc_pt_utility(gain, loss, prob_win, prob_loss, theta, Lambda, alpha, beta, gamma)
        probability = calc_prob_invest_data(invest,utility,tau, error)
        return probability
    
    
    def lognormal_pdf(value, mu, sigma):
        ''' Function to calculate the lognormal density'''
        shape  = sigma
        loc    = 0
        scale  = np.exp(mu)
        return lognorm.pdf(value, shape, loc, scale)

    def normal_pdf(value, mu, sigma):
        ''' Function to calculate the lognormal density'''
        loc    = 0
        scale  = sigma
        return norm.pdf(value, loc, scale)



    def log_likelihood_subject(param_list, sub_df, distro_estimates):
        ''' This is the functions that is going to minimized'''
        # initialize and collect
        FACTOR = 1/100
        df = sub_df.copy(True)
        parameters = {}
        lognormal_values = {}
        
        # PARAMETERS
        for idx, param in enumerate(['theta', 'Lambda', 'tau', 'alpha', 'gamma', 'error']):
            parameters[param] = param_list[idx]
        # parameters['tau'] = .28    
        parameters['beta'] = .88
        
        # calculate the likelihood of the data given the parameters
        df['ll_data'] = df.apply(lambda x: likelihood_data(x['invest'], 
                                                        x['gain']*FACTOR, 
                                                        x['loss']*FACTOR,
                                                        x['prob_win'], 
                                                        x['prob_loss'],
                                                            parameters['theta'], 
                                                            parameters['Lambda'], 
                                                            parameters['tau'], 
                                                            parameters['alpha'], 
                                                            parameters['beta'], 
                                                            parameters['gamma'],
                                                            parameters['error']), axis=1)

        # calculate the lognormal distribution weights
        for par in ['theta', 'Lambda', 'tau', 'alpha', 'gamma']:
            lognormal_values[par] = lognormal_pdf(parameters[par], distro_estimates['mu'][par], distro_estimates['sigma'][par])
        normal_values = normal_pdf(parameters['error'], 0, distro_estimates['sigma']['error'])
        # take the log of the values and sum it
        return -(np.log(df['ll_data']).sum() + np.log(lognormal_values['theta']) + np.log(lognormal_values['Lambda']) + np.log(lognormal_values['tau']) + np.log(lognormal_values['alpha']) + np.log(lognormal_values['gamma']) + np.log(normal_values))


    estimates = minimize(log_likelihood_subject, x0, args=(df, distro_estimates), method='nelder-mead', options={'xatol': 1e-8, 'disp': True, 'maxfev':1e5})
    
    return estimates
