import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.special as scp
from itertools import product
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from classes import Agent, MultiArmedBandit, World
np.set_printoptions(suppress=True)

def run_simulation(h=1000, 
				   n_train = 20,
				   n_extinction = 5, 
				#    rep=1, 
				   na=3, 
				   nb=2, 
				   ns=3, 
				#    no=3, 
				   nr=4, 
				   nc=2, 
				   T=2,
				   TAU = 10,
				   approx_pred_pol = True, 
				   approx_pred_rew = True,
				   utility = None,
				   Rho = None, 
				   debug=False, 
				   context_volatility=0.98, 
				   habit_exception=None,
				   rho_noncontingent = 0.0,
				   reward_mapping = {"outcome_1":0, "outcome_2":1, "no_reward":2, "initial_state":3},
				   action_mapping = {"lever_press":0, "no_lever_press":1},
				   training_protocol = None,
				   **pars):
	
	
	npi = na**(T-1)
	policies = np.array(list(product( np.arange(na), repeat= T-1)))

	'''       define p(s_t|s_t-1, a_t-1)      '''
	prior_states = np.array([0]*na + [1])
	state_transition_matrix = np.zeros([ns,ns,na])
	for a in range(na):
		state_transition_matrix[a,:,a] = 1

	'''          define p(o_t|s_t)            '''
	observation_generation_matrix = np.eye(ns)
	
	'''           define p(r|s,c)             '''
	lambda_H = np.ones([nr,ns])
	lambda_H[-1,-1] = 100
	counts_prior_rewards = np.stack([lambda_H for i in range(nc)],axis=-1)
	counts_prior_rewards += np.random.uniform(low=0,high=0.2, size = counts_prior_rewards.shape)
	
	prior_rewards = counts_prior_rewards / counts_prior_rewards.sum(axis=0)

	'''              define p(pi|c)           '''
	counts_prior_policies = np.zeros([npi,nc]) + h


	prior_policies = counts_prior_policies / counts_prior_policies.sum(axis=0)


	'''       define prior over contexts p(c) '''
	prior_context = np.array([0.99] + [1-0.99]*(nc-1))

	'''   p(c_t|c_t-1)'''
	p = 0.9
	q = 1-p
	q = (1-context_volatility)/(nc-1)
	context_transition_matrix = np.eye(nc)*(1-nc*q) + q
	assert(np.all(context_transition_matrix.sum(axis=0) == 1))

	'''   define Env reward generation matrix if not passed as argument'''

	Rho = Rho[:,:,training_protocol]


	# Run Simulations
	env = MultiArmedBandit(state_transition_matrix,
						Rho, 
						TAU=TAU,
						T=T,
						n_bandits = na,
						training_protocol=training_protocol,
						observation_generation_matrix=observation_generation_matrix,
						no=ns,
						rho_noncontingent = rho_noncontingent,
						reward_mapping = reward_mapping,
						action_mapping = action_mapping)

	agent = Agent(
			state_transition_matrix,
			utility,
			policies,
			prior_rewards,
			counts_prior_rewards,
			prior_policies,
			counts_prior_policies,
			prior_states,
			prior_context,
			na,
			nc,
			env,
			approx_pred_pol = approx_pred_pol,
			approx_pred_rew = approx_pred_rew,
			observation_generation_matrix = observation_generation_matrix,
			debug=debug,
			context_transition_matrix = context_transition_matrix,
			habit_exception=habit_exception
			)

	world = World(agent, env)
	world.agent.context_volatility = context_volatility
	world.agent.h = h
	world.simulate_experiment()
	
	# state_trans_matrix = world.agent.state_transition_matrix[:,:,-1,0]
	prior_rewards_matrix = world.agent.prior_rewards[-1,-1,:,:,:]
	policy_prior_matrix = world.agent.prior_policies[-1,-1,:,:]

	return world, prior_rewards_matrix, policy_prior_matrix


def create_long_df(world, pars):

	# unpack parameters
	rep = pars["rep"]
	group = pars["group"]
	sim = pars["sim"] 
	TAU = pars["TAU"] 
	T = pars["T"] 
	n_blocks_train = pars["n_blocks_train"] 
	n_blocks_extinction = pars["n_blocks_extinction"] 
	block_length = pars["block_length"]

	# create dataframe arrays
	training_protocol = pars["training_protocol"]
	agent = world.agent
	run_arr = np.full(TAU * T, rep)                                        # given parameter simulation counter  
	tau_arr = np.repeat(np.arange(TAU), T)                                 # trial 
	t_arr = np.tile(np.arange(T), TAU)                                     # time point
	observation_arr = world.environment.observations.flatten()             # state observation
	reward_arr = world.environment.rewards[:,-1].repeat(2)                 # reward received at t=1 (saved for both t=0 and t=1) 
	action_arr = np.full((TAU,T), np.nan)                                  # executed action
	action_arr[:,0] = world.environment.actions[:,0]
	action_arr = action_arr.flatten()
	h_arr = np.full(TAU * T, agent.h)                                      # habitual tendency free parameter
	context_volatility_arr = np.full(TAU * T, agent.context_volatility)    # context volatility free parameter
	experimental_group_arr = np.full(TAU * T, group)                       # experimental group: 0 - outcome 2 devalued, 1- outcome 1 devalued
	sim_arr = np.full(TAU*T,sim)                                           # overall simulation counter
	true_context = np.array(training_protocol).repeat(2)                   # true context: 0 - training, 1 - extinction
	block_arr = np.arange(n_blocks_extinction+n_blocks_train).repeat(block_length*T)   # trial block number

	# Policy / context arrays
	prior_arr = {}
	likelihood_arr = {}
	posterior_arr = {}
	context_post_arr = {}
	policy_ent_arr = {}
	policy_surpr_arr = {}
	outcome_surprise_arr = {}
	policy_post_arr = {}
	reward_prior_arr = {}

	for c in range(agent.nc):
		context_post_arr[f"posterior_context_c{c}"] = agent.posterior_context[:,:,c].flatten()
		policy_ent_arr[f"policy_entropy_c{c}"] = agent.policy_entropy[:,:,c].flatten()
		policy_surpr_arr[f"policy_surprise_c{c}"] = agent.policy_surprise[:,:,c].flatten()
		outcome_surprise_arr[f"outcome_surprise_c{c}"] = agent.outcome_surprise[:,:,c].flatten()
		for pi in range(agent.npi):
			reward_prior_arr[f"reward_s{pi}_c{c}"] = agent.prior_rewards[:,:,1,pi,c].flatten()
			prior_arr[f"prior_pi{pi}_c{c}"] = agent.prior_policies[:,:,pi,c].flatten()
			likelihood_arr[f"likelihood_pi{pi}_c{c}"] = agent.likelihood_policies[:,:,pi,c].flatten()
			posterior_arr[f"posterior_pi{pi}_c{c}"] = agent.posterior_policies[:,:,pi,c].flatten()
	
	action_likelihood = np.einsum("dtpc,dtc-> dtp", agent.posterior_policies, agent.posterior_context)
	for pi in range(agent.npi):
		policy_post_arr[f"posterior_pi{pi}"] = action_likelihood[:,:,pi].flatten()


	# Fit sigmoid/exponential to posterior over contexts and policies 

	sigmoid = lambda t, a, b, c, d: d + a / (1 + np.exp(-b*(t-c)))
	exponential = lambda t, b, c, d: d + np.exp(b*(t-c))
	fitting_functions = [sigmoid, exponential]
	t = np.arange(1,TAU+1)
	q_context = agent.posterior_context[:,1,1]
	q_action =  np.einsum("tpc,tc -> tp ", agent.posterior_policies[:,0,:,:], agent.posterior_context[:,0])[:,1]
	switch = n_blocks_train*block_length + block_length//2

	try:
		context_params, context_pcov = curve_fit(sigmoid, t, q_context, p0=[1.,1.,switch,0.])#, sigma=[0.5]*200)#*40+[2]*100+[0.5]*50)
		context_func = 0.
	except RuntimeError:
		try:
			context_pcov = np.eye(4)
			context_params = np.zeros(4)
			context_params[1:], context_pcov[1:,1:] = curve_fit(exponential, t, q_context, p0=[1.,switch,0.])#, sigma=[0.5]*40+[2]*100+[0.5]*50)
			context_params[0] = 1
			context_func = 1
		except RuntimeError:
			context_params[:] = np.nan
			context_func = np.nan
	try:
		policy_params, policy_pcov = curve_fit(sigmoid, t[10:], q_action[10:], p0=[1.,1.,switch,0.])#, sigma=[0.5]*40+[2]*100+[0.5]*50)
		policy_func = 0
	except RuntimeError:
		try:
			policy_params = np.zeros(4)
			policy_params[1:], pcov = curve_fit(exponential, t[10:], q_action[10:], p0=[1.,switch,0.])#, sigma=[0.5]*40+[2]*100+[0.5]*50)
			policy_params[0] = 1.
			policy_func = 1
		except RuntimeError:
			policy_params[:] = np.nan
			policy_func = np.nan


	# create dataframe dictionary
	long_df_dict = {
		"run": run_arr,
		"tau": tau_arr,
		"t": t_arr,
		"observation": observation_arr,
		"action": action_arr,
		"reward": reward_arr,
		"h": h_arr,
		"context_volatility": context_volatility_arr,
		"experimental_group": experimental_group_arr,
		"phase": np.array(["training","extinction"])[training_protocol].repeat(T),
		"sim":sim_arr,
		"true_context":true_context,
		"block":block_arr,
		"context_a": np.array([context_params[0]]*(TAU*T)),
		"context_b": np.array([context_params[1]]*(TAU*T)),
		"context_c": np.array([context_params[2]]*(TAU*T)),
		"context_d": np.array([context_params[3]]*(TAU*T)),
		**prior_arr,
		**likelihood_arr,
		**posterior_arr,
		**context_post_arr,
		**policy_ent_arr,
		**policy_surpr_arr,
		**outcome_surprise_arr,
		**policy_post_arr,
		**reward_prior_arr
	}
	return pd.DataFrame(long_df_dict)


def fit_posterior_traces(q_context, q_action, switch, TAU):
    """Fit sigmoid/exponential functions to context and policy posterior traces."""

    sigmoid = lambda t, a, b, c, d: d + a / (1 + np.exp(-b * (t - c)))
    exponential = lambda t, b, c, d: d + np.exp(b * (t - c))

    t = np.arange(1, TAU + 1)

    try:
        context_params, context_pcov = curve_fit(sigmoid, t, q_context, p0=[1., 1., switch, 0.])
        context_func = 0
    except RuntimeError:
        try:
            exponential_params, exponential_pcov = curve_fit(exponential, t, q_context, p0=[1., switch, 0.])
            context_params = np.zeros(4)
            context_params[0] = 1.
            context_params[1:] = exponential_params
            context_pcov = np.eye(4)
            context_pcov[1:, 1:] = exponential_pcov
            context_func = 1
        except RuntimeError:
            context_params = np.full(4, np.nan)
            context_pcov = np.full((4, 4), np.nan)
            context_func = np.nan

    try:
        policy_params, policy_pcov = curve_fit(sigmoid, t[10:], q_action[10:], p0=[1., 1., switch, 0.])
        policy_func = 0
    except RuntimeError:
        try:
            exponential_params, policy_pcov = curve_fit(exponential, t[10:], q_action[10:], p0=[1., switch, 0.])
            policy_params = np.zeros(4)
            policy_params[0] = 1.
            policy_params[1:] = exponential_params
            policy_func = 1
        except RuntimeError:
            policy_params = np.full(4, np.nan)
            policy_pcov = np.full((3, 3), np.nan)
            policy_func = np.nan

    se = np.sqrt(np.diag(context_pcov)) if np.all(np.isfinite(context_pcov)) else np.full(4, np.nan)

    return context_params, context_pcov, context_func, policy_params, policy_func, se
