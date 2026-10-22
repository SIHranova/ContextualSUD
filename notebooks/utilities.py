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
np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
np.seterr(over='ignore', invalid='ignore')



def plot_reward_regime(k_range, switch=100, TAU=400, start=0.9, stop=0.1,dpi=100, env_order=["low_certainty", "high_certainty"]):

# 	# function that ranges between start and stop decays with a varying rate k
	logistic = lambda x, k: stop - (start-stop) / (np.exp(switch*k) - 1) + (np.exp(switch*k) + 1) / (np.exp(switch*k) - 1)*(start-stop)/(1+np.exp(k*(x-switch)))
	plt.figure(figsize=(4,3),dpi=dpi)
	style = ['solid','dashed']

	for ki, k in enumerate(k_range):
		x = np.arange(0,switch*2,1)
		p_win = logistic(x,k)
		plt.plot(np.arange(TAU), p_win, 'k', linestyle=style[ki], label=env_order[ki])
		plt.xlabel("trial", fontsize=16)
		plt.ylabel("Reward preference", fontsize=16)
		plt.ylabel(r"$\bar{p}(o=\text{reward})$", fontsize=16)

		plt.ylim([-0.1,1.1])
		plt.legend(bbox_to_anchor=(1,0.7))


def run_simulation(h=1000,
                   na=3,
                   npi=None,
                   ns=3,
                   nr=4,
                   nc=2,
                   T=2,
                   TAU=10,
                   approx_pred_pol=True,
                   approx_pred_rew=True,
                   utility=None,
                   Rho=None,
                   debug=False,
                   context_volatility=0.98,
                   habit_exception=None,
                   rho_noncontingent=0.0,
                   reward_mapping={"outcome_1": 0, "outcome_2": 1, "no_reward": 2, "initial_state": 3},
                   action_mapping={"lever_press": 0, "no_lever_press": 1},
                   training_protocol=None,
                   learn_cues = False,
                   prior_color_cue = None,
                   color_cue_context_mapping = None,
                   **pars):

    npi = na**(T-1)
    policies = np.array(list(product(np.arange(na), repeat=T-1)))

    prior_states = np.array([0]*na + [1])
    state_transition_matrix = np.zeros([ns, ns, na])
    for a in range(na):
        state_transition_matrix[a, :, a] = 1

    observation_generation_matrix = np.eye(ns)

    lambda_H = np.ones([nr,ns])
    lambda_H[-1,-1] = 100
    counts_prior_rewards = np.stack([lambda_H for i in range(nc)],axis=-1)
    counts_prior_rewards += np.random.uniform(low=0,high=0.2, size = counts_prior_rewards.shape)
	
    prior_rewards = counts_prior_rewards / counts_prior_rewards.sum(axis=0)

    counts_prior_policies = np.zeros([npi, nc]) + h
    prior_policies = counts_prior_policies / counts_prior_policies.sum(axis=0)

    '''       define prior over contexts p(c) '''
    prior_context = np.array([0.99] + [1-0.99]*(nc-1))

    q = (1-context_volatility)/(nc-1)
    context_transition_matrix = np.eye(nc)*(1-nc*q) + q
    assert(np.all(context_transition_matrix.sum(axis=0) == 1))

    if Rho.shape[-1] == TAU:
        trial_Rho = Rho
    elif Rho.shape[-1] == nc:
        trial_Rho = Rho[:, :, training_protocol]
    else:
        raise ValueError("Rho must have one slice per context or trial")

    env = MultiArmedBandit(state_transition_matrix,
                           trial_Rho,
                           TAU=TAU, 
                           T=T, 
                           n_bandits=na, 
                           training_protocol=training_protocol, 
                           observation_generation_matrix=observation_generation_matrix,
                           no=ns, 
                           rho_noncontingent=rho_noncontingent, 
                           reward_mapping=reward_mapping, 
                           action_mapping=action_mapping,
                           color_cue_context_mapping=color_cue_context_mapping)

    agent = Agent(state_transition_matrix,
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
                  approx_pred_pol=approx_pred_pol,
                  approx_pred_rew=approx_pred_rew,
                  observation_generation_matrix=observation_generation_matrix,
                  debug=debug,
                  context_transition_matrix=context_transition_matrix,
                  habit_exception=habit_exception,
                  learn_cues=learn_cues,
                  prior_color_cue=prior_color_cue)

    world = World(agent, env)
    world.agent.context_volatility = context_volatility
    world.agent.h = h
    world.simulate_experiment()

    prior_rewards_matrix = world.agent.prior_rewards[-1, -1, :, :, :]
    policy_prior_matrix = world.agent.prior_policies[-1, -1, :, :]

    return world, prior_rewards_matrix, policy_prior_matrix

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


def create_long_df_multiple_levers(world, pars):

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

	# fit posterior traces
	q_context = agent.posterior_context[:,1,1]
	q_action =  np.einsum("tpc,tc -> tp ", agent.posterior_policies[:,0,:,:], agent.posterior_context[:,0])[:,1]
	switch = n_blocks_train*block_length + block_length//2
	context_params, _, _, _, _, _ = fit_posterior_traces(q_context, q_action, switch, pars["TAU"])

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


def create_long_df_gradual_change(world, p, environment_mapping, agent_mapping):

    TAU = p["TAU"]
    T = p["T"]
    training_protocol = p["training_protocol"]
    agent = world.agent
    reward_decay_rate_arr = np.full(TAU * T, None)
    utility_decay_rate_arr = np.full(TAU * T, None)

    if p["changing_rewards"]:
        reward_decay_rate_arr = np.full(TAU * T, p["decay_rate"])
        utility_0_arr = p["utility"][0].repeat(TAU*T)
        utility_1_arr = p["utility"][1].repeat(TAU*T)
        utility_2_arr = p["utility"][2].repeat(TAU*T)
    elif p["changing_utility"]:
        utility_decay_rate_arr = np.full(TAU * T, p["decay_rate"])
        utility_0_arr = p["utility"][:,0].repeat(T)
        utility_1_arr = p["utility"][:,1].repeat(T)
        utility_2_arr = p["utility"][:,2].repeat(T)


    run_arr = np.full(TAU*T, p.get("rep", p.get("sim", 0)))
    tau_arr = np.repeat(np.arange(TAU), T)
    t_arr = np.tile(np.arange(T), TAU)
    observation_arr = world.environment.observations.flatten()
    reward_arr = world.environment.rewards.flatten()
    action_arr = np.full((TAU, T), np.nan)
    action_arr[:, 0] = world.environment.actions[:, 0]
    action_arr = action_arr.flatten()
    habitual_tendency_arr = np.full(TAU*T, 1/p["h"])
    h_arr = np.full(TAU*T, agent_mapping[p["h"]])
    context_volatility_arr = np.full(TAU*T, p["context_volatility"])
    environment_type_arr = np.full(TAU*T, environment_mapping[p["decay_rate"]])
    sim_arr = np.full(TAU*T,p["sim"])
    true_context = np.array(training_protocol).repeat(T)


    prior_arr = {}
    likelihood_arr = {}
    posterior_arr = {}
    context_post_arr = {}
    policy_ent_arr = {}
    policy_surpr_arr = {}
    outcome_surpr_arr = {}
    policy_post_arr = {}
    reward_prior_arr = {}

    for c in range(agent.nc):
        context_post_arr[f"posterior_context_c{c}"] = agent.posterior_context[:, :, c].flatten()
        policy_ent_arr[f"policy_entropy_c{c}"] = agent.policy_entropy[:, :, c].flatten()
        policy_surpr_arr[f"policy_surprise_c{c}"] = agent.policy_surprise[:, :, c].flatten()
        outcome_surpr_arr[f"outcome_surprise_c{c}"] = agent.outcome_surprise[:, :, c].flatten()

        for pi in range(agent.npi):
            reward_prior_arr[f"reward_s{pi}_c{c}"] = agent.prior_rewards[:, :, 1, pi, c].flatten()
            prior_arr[f"prior_pi{pi}_c{c}"] = agent.prior_policies[:, :, pi, c].flatten()
            likelihood_arr[f"likelihood_pi{pi}_c{c}"] = agent.likelihood_policies[:, :, pi, c].flatten()
            posterior_arr[f"posterior_pi{pi}_c{c}"] = agent.posterior_policies[:, :, pi, c].flatten()

    action_likelihood = np.einsum("dtpc,dtc->dtp", agent.posterior_policies, agent.posterior_context)
    for pi in range(agent.npi):
        policy_post_arr[f"posterior_pi{pi}"] = action_likelihood[:, :, pi].flatten()

    q_context = agent.posterior_context[:, 1, 1]
    q_action = np.einsum("tpc,tc -> tp", agent.posterior_policies[:, 0, :, :], agent.posterior_context[:, 0])[:, 1]

    context_params, context_pcov, context_func, policy_params, policy_func, se = fit_posterior_traces(q_context, q_action, p["switch"], TAU)

    context_a_arr = np.full(TAU*T, context_params[0])
    context_b_arr = np.full(TAU*T, context_params[1])
    context_c_arr = np.full(TAU*T, context_params[2] - p["switch"])
    context_d_arr = np.full(TAU*T, context_params[3])
    context_func_arr = np.full(TAU*T, context_func)
    policy_a_arr = np.full(TAU*T, policy_params[0])
    policy_b_arr = np.full(TAU*T, policy_params[1])
    policy_c_arr = np.full(TAU*T, policy_params[2] - p["switch"])
    policy_d_arr = np.full(TAU*T, policy_params[3])
    policy_func_arr = np.full(TAU*T, policy_func)

    long_df_dict = {
        "sim": sim_arr,
        "environment_type": environment_type_arr,
        "utility_decay_rate": utility_decay_rate_arr,
        "reward_decay_rate": reward_decay_rate_arr,
        "h": h_arr,
        "habitual_tendency": habitual_tendency_arr,
        "run": run_arr,
        "true_context": true_context,
        "tau": tau_arr,
        "t": t_arr,
        "observation": observation_arr,
        "action": action_arr,
        "reward": reward_arr,
        "context_volatility": context_volatility_arr,
        "utility_0": utility_0_arr,
        "utility_1": utility_1_arr,
        "utility_2": utility_2_arr,
        **prior_arr,
        **likelihood_arr,
        **posterior_arr,
        **context_post_arr,
        **policy_ent_arr,
        **policy_surpr_arr,
        **outcome_surpr_arr,
        **policy_post_arr,
        **reward_prior_arr,
        "context_a": context_a_arr,
        "context_b": context_b_arr,
        "context_c": context_c_arr,
        "context_d": context_d_arr,
        "context_func": context_func_arr,
        "policy_a": policy_a_arr,
        "policy_b": policy_b_arr,
        "policy_c": policy_c_arr,
        "policy_d": policy_d_arr,
        "policy_func": policy_func_arr,
    }

    return pd.DataFrame(long_df_dict)
