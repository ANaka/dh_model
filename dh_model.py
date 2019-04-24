from brian2 import *
from brian2.units.fundamentalunits import get_unit_for_display

class adex_group(NeuronGroup):
	def __init__(self, name, N = 1):
		model = '''dv/dt = (Delta_T*g_l*exp((-V_th + v)/Delta_T)
				 + g_e*(E_e - v) + g_i*(E_i - v) + g_l*(E_l - v) - u + I_ex(t, i))/C_m : volt (unless refractory)
				 dg_e/dt = -g_e/tau_e : siemens
				 dg_i/dt = -g_i/tau_i : siemens
				 du/dt = (alpha*(-E_l + v) - u)/tau_u : amp
				 g_l : siemens (constant)'''
		reset_eq = '''v = V_r
		u += b'''
		threshold_eq = 'v > V_th'
		refractory_eq = 'tau_r'
		super().__init__(
			N = N, model = model,
			reset = reset_eq,
			threshold = threshold_eq,
			refractory = refractory_eq,
			method = 'euler',
			name = name)
		
		
	def reset_variables(self):
		self.g_i = '0 * nS'
		self.g_e = '0 * nS'
		self.v = self.namespace['E_l']  
		self.u = '0 * pA'
	
	def initialize_I_ex_array(self,duration,sim_dt):
		tb = np.arange(0,duration,sim_dt)
		I_ex_mu = self.namespace['I_ex_mu']
		I_ex_sigma = self.namespace['I_ex_sigma']
		I_ex_template = (I_ex_mu + I_ex_sigma * randn(self.N))
		I_ex_array = np.tile(I_ex_template, (tb.shape[0],1))
		self.namespace['I_ex_array'] = np.array(I_ex_array)
		self.namespace['I_ex'] = TimedArray(I_ex_array,sim_dt)
	
	def set_I_ex_from_array(self,I_ex_array,sim_dt):
		assert type(I_ex_array) is numpy.ndarray
		tb = np.arange(0,len(I_ex_array))*sim_dt
		self.namespace['I_ex_array'] = I_ex_array
		self.namespace['I_ex'] = TimedArray(I_ex_array * pA,sim_dt)
	
	def add_I_ex_step(self,duration, sim_dt, start, stop, amplitude, indices = None):
		tb = np.arange(0,duration,sim_dt)
		I_ex_template = np.zeros((len(tb),self.N))
		
		if not indices:
			indices = range(self.N)        
		I_ex_template = self.namespace['I_ex_array'] * pA
		
		for t_ind,t in enumerate(tb):
			if (t >= start) & (t < stop):
				for i in indices:
					I_ex_template[t_ind,i] = amplitude
		self.namespace['I_ex_array'] = np.array(I_ex_template)
		self.namespace['I_ex'] = TimedArray(I_ex_template,sim_dt)

	def set_cell_type(self, cell_type):
		params = dp.get_neuron_params(cell_type)
		
		self.outputs = params['outputs']
		
		for key, value in params['intrinsics'].items():
			self.namespace[key] = value
			
		# constant parameters that vary between neurons in this group
		g_l_mu = params['intrinsics']['g_l_mu']
		g_l_sigma = params['intrinsics']['g_l_sigma']
		self.g_l = (g_l_mu + g_l_sigma * randn(self.N))
		self.reset_variables()
		
	def set_intrinsics(self,intrinsics):
		
		for key, value in intrinsics.items():
			self.namespace[key] = value
			
		# constant parameters that vary between neurons in this group
		g_l_mu = intrinsics['g_l_mu']
		g_l_sigma = intrinsics['g_l_sigma']
		self.g_l = (g_l_mu + g_l_sigma * randn(self.N))
		self.reset_variables()
		
	def set_intrinsics_from_dfs(self,intrinsics_df):
		params = intrinsics_df.columns
		intrinsics = {}
		for param in params:
			tempstr = str(intrinsics_df.loc[self.name,param]) + ' * ' + intrinsics_df.loc['units',param]
			intrinsics[param] = eval(tempstr)
		
		for key, value in intrinsics.items():
			self.namespace[key] = value
			
		# constant parameters that vary between neurons in this group
		g_l_mu = intrinsics['g_l_mu']
		g_l_sigma = intrinsics['g_l_sigma']
		self.g_l = (g_l_mu + g_l_sigma * randn(self.N))
		self.reset_variables()
	
	def set_outputs_from_dfs(self,output_dfs):
		temp_dict ={}
		outputs = {}
		for param_name,output_df in output_dfs.items():
			temp_dict[param_name] = output_df.loc[self.name,:]
		
		targets = list(temp_dict['p'][temp_dict['p']>0].index)
		for post in targets:
			outputs[post] = {}
			for param_name,param in temp_dict.items():
				if param_name in ['w_mu','w_sigma']:
					outputs[post][param_name] = param[post] * siemens
				else:
					outputs[post][param_name] = param[post]
		self.namespace['outputs'] = outputs        
		
class afferent_group(SpikeGeneratorGroup):
	def __init__(self, name, N = 1):
		super().__init__(N = N,indices = [0], times = [9999*second],name = name)
			
def connect_all_ngs_from_dfs(ngs,ng_output_dfs):
	s = {}
	for pre_name,pre in ngs.items():
		for post_name,post in ngs.items():
			if ng_output_dfs['p'].loc[pre_name,post_name]>0:
				temp_dict ={}
				for param_name,output_df in ng_output_dfs.items():
					if param_name in ['w_mu','w_sigma']:
						temp_dict[param_name] = output_df.loc[pre_name,post_name] * siemens
					else:
						temp_dict[param_name] = output_df.loc[pre_name,post_name]
				
				temp_s = connect_neuron_groups(pre, post,
					**temp_dict)
				s[temp_s.name]= temp_s  
	return s

def connect_all_ags_from_dfs(ngs,ags,ag_output_dfs):
	s = {}
	for pre_name,pre in ags.items():
		for post_name,post in ngs.items():
			if ag_output_dfs['p'].loc[pre_name,post_name]>0:
				temp_dict ={}
				for param_name,output_df in ag_output_dfs.items():
					if param_name in ['w_mu','w_sigma']:
						temp_dict[param_name] = output_df.loc[pre_name,post_name] * siemens
					else:
						temp_dict[param_name] = output_df.loc[pre_name,post_name]
				
				temp_s = connect_neuron_groups(pre, post,
					**temp_dict)
				s[temp_s.name]= temp_s
	return s
	


def connect_neuron_groups(pre_group, post_group,
						  conductance_name = 'e',
						  p = 1, 
						  w_mu = 1 * nS, 
						  w_sigma = 0.1 * nS):
	on_pre_equation = 'g_' + conductance_name + '_post += w'
	name = pre_group.name + '_to_'+post_group.name
	S = Synapses(pre_group, post_group,model = 'w : siemens',
							 on_pre = on_pre_equation,name = name)
	S.connect(p = p)
	S.w =  w_mu + w_sigma * randn(S.w.shape[0])
	S.w['w < 0 * nS'] = 0 * nS
	return S

def output_dfs_2_dict(output_dfs,nts):   
	outputs = {}
	for pre in nts:
		for post in nts:
			outputs[pre][post]={}
		
	for param_name, output_df in output_dfs.items():
		for pre in nts:
			for post in output_df.columns:
				outputs[pre][post]={}
				outputs[pre][post][param_name] = output_df.loc[pre,post]
	return outputs

def generate_periodic_spike_times(rate, start_time, stop_time):
	###unitless but do it in seconds to make it easy
	duration = stop_time - start_time
	num_spikes = np.floor(duration * rate)
	true_duration = num_spikes / rate
	spike_times = np.linspace(start_time, start_time + true_duration, num_spikes)
	return spike_times

def gaussian_psth(duration,sim_dt,mu = 0 * ms, sigma = 0 * ms):
		time_bins = np.arange(0,duration,sim_dt)
		mu_ = mu 
		sigma_ = sigma  
		psth = (1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (time_bins - mu)**2 / (2 * sigma**2)))
		psth = psth/sum(psth)
		return psth,time_bins
	
def uniform_psth(duration,sim_dt,):
		time_bins = np.arange(0,duration,sim_dt)
		psth = np.ones(time_bins.shape)*sim_dt/second
		return psth,time_bins

def spikes_from_psth(psth):
	return np.random.binomial(1,psth)

def generate_population_spikes(kernel,rates,sim_dt):
	spikes = []
	ind = []
	for i,r in enumerate(rates):
		this_spikes = np.where(spikes_from_psth(kernel*r))[0]
		this_ind = np.ones(this_spikes.shape) * i
		spikes.append(this_spikes)
		ind.append(this_ind)
	spikes = np.concatenate(spikes) *sim_dt
	ind = np.concatenate(ind).astype(int)
	return spikes, ind      

def initialize_inputs(ags, ngs, num_trials, trial_durations ):
	# if not trial_durations:
	#     trial_durations = [duration] * num_trials
	
	inputs = {}
	for trial in range(num_trials):
		inputs[trial] = {}
		for at,ag in ags.items():
			inputs[trial][at] = {}
			inputs[trial][at]['spike_times'] = ag._spike_time
			inputs[trial][at]['indices'] = ag._neuron_index
		for nt,ng in ngs.items():
			inputs[trial][nt] = {}
			inputs[trial][nt]['I_ex'] = ng.namespace['I_ex_array']
		inputs[trial]['trial_duration'] = trial_durations[trial]
	return inputs

def concat_inputs(inputs,sim_objects):
	I_ex_arrays ={}
	for nt in sim_objects['neuron_groups'].keys():
		I_ex_arrays[nt]=[]
		for trial,this_input in inputs.items():
			I_ex_arrays[nt].append(this_input[nt]['I_ex'])
		I_ex_arrays[nt] = np.concatenate(I_ex_arrays[nt])
	#need to test    
	ag_inputs_full_trial = {}
	for at in sim_objects['afferent_groups'].keys():
		ag_inputs_full_trial[at] = {}
		t_start = 0 * second
		ag_inputs_full_trial[at]['spike_times'] =[]
		ag_inputs_full_trial[at]['indices'] = []
		for trial,this_input in inputs.items():
			ag_inputs_full_trial[at]['spike_times'].append(this_input[at]['spike_times'] + t_start)
			ag_inputs_full_trial[at]['indices'].append(this_input[at]['indices'])
			t_start += this_input['trial_duration']
		ag_inputs_full_trial[at]['spike_times'] =np.concatenate(ag_inputs_full_trial[at]['spike_times'])*second
		ag_inputs_full_trial[at]['indices'] =np.concatenate(ag_inputs_full_trial[at]['indices'])
	return I_ex_arrays,ag_inputs_full_trial

def make_active_inds_binary(frac_active, N):
	num_active = np.round(frac_active * N).astype(int)
	active_inds = np.random.choice(np.array(range(N)),size = num_active,replace = False)
	active_inds_binary = np.zeros((N))
	for ind in active_inds:
		active_inds_binary[ind] = 1
	return active_inds, active_inds_binary

# should make outputs and cost function together

def get_trialwise_spikes(inputs, sim_objects):
	observed_spikes = {}
	trial_start = 0 * ms
	for trial, this_input in inputs.items():
		observed_spikes[trial] = {}
		trial_duration = this_input['trial_duration']
		trial_end = trial_start + trial_duration
		for nt in sim_objects['spike_mons'].keys():
			N = sim_objects['neuron_groups'][nt].N
			observed_spikes[trial][nt]={}
			observed_spikes[trial][nt]['counts'] = []
			observed_spikes[trial][nt]['rates'] = []
			spike_trains = sim_objects['spike_mons'][nt].spike_trains()
			for neuron,spike_times in spike_trains.items():
				current_trial_spike_times = spike_times[(spike_times>=trial_start) & (spike_times<trial_end)]
				current_trial_spike_count = len(current_trial_spike_times)
				observed_spikes[trial][nt]['counts'].append(current_trial_spike_count)
				observed_spikes[trial][nt]['rates'].append(current_trial_spike_count/trial_duration*second)
				
			observed_spikes[trial][nt]['mean_count'] = mean(observed_spikes[trial][nt]['counts'])
			observed_spikes[trial][nt]['mean_rate'] = mean(observed_spikes[trial][nt]['rates'])
		trial_start += trial_duration
	return observed_spikes

def mean_spike_count_cost_func(target_outputs, observed_spikes):
	differences = []
	for trial, this_trial_output in target_outputs.items():
		for nt, target_counts in this_trial_output.items():
			differences.append((target_counts - observed_spikes[trial][nt]['mean_count'])**2)
	cost = sum(differences)**0.5
	return cost

def initialize_target_outputs(inputs):
	target_outputs = {}
	for trial,this_input in inputs.items():
		target_outputs[trial] = {}
	return target_outputs   

def add_target_mean_rate(target_outputs, rate_vec, nt):
	assert len(rate_vec) == len(target_outputs)
	for trial,rate in enumerate(rate_vec):
		target_outputs[trial][nt] = rate
	return target_outputs

def wrap_update_w(syn):
	def update_w(w_mu):
		w_sigma = 0.1
		syn.w = (w_mu + w_sigma * randn(syn.w.shape[0]))* nS
		syn.w['w < 0 * nS'] = 0 * nS
	new_func = update_w
	return new_func

def gen_s_w_update_funcs(s,s_names = None):
	s_w_update_funcs = {}
	if not s_names:
		s_names = list(s.keys())
	for s_name in s_names:
		s_w_update_funcs[s_name] = wrap_update_w(s[s_name])
	return s_w_update_funcs

def wrap_update_intrinsics(ng, var_name):
	ng
	# def update_intrinsics(x):
	# 	this_unit = ng.namespace[var_name].in_best_unit(python_code= True).split()[-1]
	# 	temp_str = str(x) +'* ' +this_unit
	# 	ng.namespace[var_name] = eval(temp_str)
	def update_intrinsics(x):
            this_unit = get_unit_for_display(ng.namespace[var_name].dim)
            temp_str = str(x) + '*' + this_unit
            ng.namespace[var_name] = eval(temp_str)
	new_func = update_intrinsics
	return new_func

def gen_intrinsics_update_funcs(ngs,ng_var_pairs_list):
	intrinsics_update_funcs = {}
	for ng_name,var_name in ng_var_pairs_list:
		intrinsics_update_funcs[(ng_name,var_name)] = wrap_update_intrinsics(ngs[ng_name],var_name)
	return intrinsics_update_funcs


	

def run_sim(sim_objects, inputs, target_outputs, cost_func):
	# inputs: dictionary with fields 'ag_rates','ng_I_ex', 'num_trials'
	# sim_objects: dictionary with fields 'network','neuron_group_list','afferent_group_list','synapse_list'
			
	I_ex_full, ag_inputs_full = concat_inputs(inputs,sim_objects)
	for nt,ng in sim_objects['neuron_groups'].items():
		ng.set_I_ex_from_array(I_ex_full[nt],sim_objects['sim_dt'])
	for at,ag in sim_objects['afferent_groups'].items():
		ag.set_spikes(indices = ag_inputs_full[at]['indices'], times = ag_inputs_full[at]['spike_times']) 
	
	num_trials = len(inputs)
	for trial in range(num_trials):
		duration = inputs[trial]['trial_duration']
		for nt,ng in sim_objects['neuron_groups'].items():
			ng.reset_variables()
#         for at,ag in ags.items():
#             spike_times = inputs[trial][at]['spike_times']
#             indices = inputs[trial][at]['indices']
#             ag.set_spikes(indices = indices, times = spike_times + net.t)
		
		sim_objects['network'].run(duration)
	observed_spikes = get_trialwise_spikes(inputs, sim_objects)
	cost = cost_func(target_outputs,observed_spikes)
	return cost, observed_spikes

def fit_network(x, update_funcs, sim_objects,  inputs, target_outputs, cost_func, opts):
	# inputs: dictionary with fields 'ag_rates','ng_I_ex'
	# sim_objects: dictionary with fields 'network','ngs','ags','s'
	# target_outputs: dictionary with fields corresponding to neuron_groups
	# cost_func: of form cost = f(spks,target_outputs)
	# free_params: dict with fields ['intrinsics','synaptic']
	
	sim_objects['network'].restore()
	beta = {}
	for param,this_x in zip(update_funcs.keys(),x):
		beta[param] = this_x
		update_funcs[param](this_x)
	cost,observed_spikes = run_sim(sim_objects, inputs, target_outputs, cost_func)
	
	if opts['track']:
		opts['tracker']['beta'].append(beta)
		opts['tracker']['observed_spikes'].append(observed_spikes)
		opts['tracker']['cost'].append(cost)
	if opts['verbose']:
		print(cost,
			  ["{:.1f}".format(observed_spikes[trial]['pr_noci']['mean_count']) for trial in range(len(inputs))],
			  ["{:.1f}".format(observed_spikes[trial]['pr_WDR']['mean_count']) for trial in range(len(inputs))])
	return cost

def initialize_bounds(update_funcs):
	# doesn't work for synapses yet
	# maybe should pull from param dfs
	uf_keys = list(update_funcs.keys())
	init_vals = {}
	for uf_key in uf_keys:
		if type(uf_key) is tuple:
			init_vals[uf_key] = sim_objects['neuron_groups'][uf_key[0]].namespace[uf_key[1]]
		elif type(uf_key) is str:
			init_vals[uf_key] = np.mean(sim_objects['synapses'][uf_key].w)
	return init_vals
