from brian2 import *
import pandas as pd
import sympy
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

def LIF_intrinsics(E_l = -70 * mV,
		   E_e = 0 * mV,
		   E_i = -70 * mV,
		   C_m = 40 * pF,
		   V_th = -40 * mV,
		   V_r = -50 * mV,
		   tau_e = 2 * ms,
		   tau_i = 5 * ms,
		   tau_r = 1 * ms,
		   g_l_mu = 2 * nS,
		   g_l_sigma = 0.25 * nS,
		   I_ex_mu = 0 * pA,
		   I_ex_sigma = 0 * pA):

		intrinsics = {}
		intrinsics['E_l'] = E_l 
		intrinsics['E_e'] = E_e 
		intrinsics['E_i'] = E_i 
		intrinsics['C_m'] = C_m 
		intrinsics['V_th'] = V_th 
		intrinsics['V_r'] = V_r
		intrinsics['tau_e'] = tau_e
		intrinsics['tau_i'] = tau_i
		intrinsics['tau_r'] = tau_r
		intrinsics['g_l_mu'] = g_l_mu
		intrinsics['g_l_sigma'] = g_l_sigma
		intrinsics['I_ex_mu'] = I_ex_mu
		intrinsics['I_ex_sigma'] = I_ex_sigma
		params['intrinsics'] = intrinsics
		
		params['dynamics'] = 'LIF'
		params['model_eq'] = Equations('''
		dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) +
				 I_ex(t,i))/C_m    : volt (unless refractory)
		dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance
		dg_i/dt = -g_i/tau_i  : siemens  # post-synaptic exc. conductance
		g_l : siemens (constant)
		''')
		params['reset_eq'] = 'v = V_r'
		params['threshold_eq'] = 'v > V_th'
		params['refractory_eq'] = 'tau_r'


def adex_intrinsics(E_l = -70 * mV,
					   E_e = 0 * mV,
					   E_i = -70 * mV,
					   C_m = 40 * pF,
					   V_th = -40 * mV,
					   V_r = -50 * mV,
					   tau_e = 2 * ms,
					   tau_i = 5 * ms,
					   tau_r = 1 * ms,
					   g_l_mu = 2 * nS,
					   g_l_sigma = 0.25 * nS,
					   I_ex_mu = 0 * pA,
					   I_ex_sigma = 0 * pA,
					   b = 8 * pA,
					   alpha = 2 * nS,
					   Delta_T = 1* mV,                 
					   tau_u = 150 * ms):

		intrinsics = {}
		intrinsics['E_l'] = E_l 
		intrinsics['E_e'] = E_e 
		intrinsics['E_i'] = E_i 
		intrinsics['C_m'] = C_m 
		intrinsics['V_th'] = V_th 
		intrinsics['V_r'] = V_r
		intrinsics['tau_e'] = tau_e
		intrinsics['tau_i'] = tau_i
		intrinsics['tau_r'] = tau_r
		intrinsics['g_l_mu'] = g_l_mu
		intrinsics['g_l_sigma'] = g_l_sigma
		intrinsics['I_ex_mu'] = I_ex_mu
		intrinsics['I_ex_sigma'] = I_ex_sigma
		intrinsics['b'] = b
		intrinsics['alpha'] = alpha
		intrinsics['Delta_T'] = Delta_T
		intrinsics['tau_u'] = tau_u
		intrinsics['tau_m'] = intrinsics['C_m'] * (1/intrinsics['g_l_mu'])
		
		params = {}
		params['intrinsics'] = intrinsics
		params['dynamics'] = 'adex'
		params['model_eq'] = Equations('''
		dv/dt = ((g_l *(E_l-v) + g_l * Delta_T * exp ((v - V_th)/Delta_T)
		- u + g_e*(E_e-v) + g_i*(E_i-v) + I_ex(t,i))/C_m) : volt (unless refractory)    
		dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance
		dg_i/dt = -g_i/tau_i  : siemens  # post-synaptic inh. conductance
		du/dt = (alpha * (v - E_l) - u)/tau_u  : amp  
		g_l : siemens (constant)
		''')
		params['reset_eq'] = '''
		v = V_r 
		u += b
		'''
		params['threshold_eq'] = 'v > V_th'
		params['refractory_eq'] = 'tau_r'
		return params

def get_neuron_params(cell_type, **kwargs):

	############################################
	# Projection neurons
	############################################

	params = adex_intrinsics()
	params['outputs'] = {}
	if cell_type == 'pr_noci':
		params['N'] = 17
		params['location'] = ['1']
		params['neurotransmitter'] = 'e'
		params['morphology'] = 'projection'
		
		# intrinsic props
		# params['intrinsics']['E_l'] = E_l 
  #       params['intrinsics']['C_m'] = C_m 
  #       params['intrinsics']['V_th'] = V_th 
  #       params['intrinsics']['V_r'] = V_r
  #       params['intrinsics']['tau_r'] = tau_r
  #       params['intrinsics']['g_l_mu'] = g_l_mu
  #       params['intrinsics']['g_l_sigma'] = g_l_sigma
  #       params['intrinsics']['I_ex_mu'] = I_ex_mu
  #       params['intrinsics']['I_ex_sigma'] = I_ex_sigma
  #       params['intrinsics']['b'] = b
  #       params['intrinsics']['alpha'] = alpha
  #       params['intrinsics']['Delta_T'] = Delta_T
  #       params['intrinsics']['tau_u'] = tau_u


		# connectivity
		
		
	elif cell_type == 'pr_pruri':
		params['N'] = 5
		params['location'] = ['1']
		params['neurotransmitter'] = 'e'
		params['morphology'] = 'projection'
		# connectivity
		

	elif cell_type == 'pr_WDR':
		params['N'] = 17
		params['location'] = ['3']
		params['neurotransmitter'] = 'e'
		params['morphology'] = '?'
		# connectivity
		
		
	############################################
	# Excitatory
	############################################
	
	elif cell_type == 'e_vertical':
		params['N'] = 338
		params['location'] = ['2o']
		params['neurotransmitter'] = 'e'
		params['morphology'] = 'vertical'
		# connectivity
		
		params['outputs']['pr_noci'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
	
	elif cell_type == 'e_tac1':
		params['N'] = 234
		params['location'] = ['2o']
		params['neurotransmitter'] = 'e'
		params['morphology'] = '?'
		# connectivity
		
		params['outputs']['pr_noci'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		

	elif cell_type == 'e_grp':
		params['N'] = 203
		params['location'] = ['2o','2id']
		params['neurotransmitter'] = 'e'
		params['morphology'] = 'central'
		# connectivity
		
		params['outputs']['e_vertical'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_grpr'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		
	elif cell_type == 'e_nts':
		params['N'] = 200
		params['location'] = ['2iv','3']
		params['neurotransmitter'] = 'e'
		params['morphology'] = 'radial'
		# connectivity
		
		params['outputs']['e_grp'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_nts'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_tac2'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}

	elif cell_type == 'e_tac2':
		params['N'] = 200
		params['location'] = ['2iv','3']
		params['neurotransmitter'] = 'e'
		params['morphology'] = 'central'
		# connectivity
		
		params['outputs']['e_grp'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_nts'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_tac2'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}


	elif cell_type == 'e_cck':
		params['N'] = 90
		params['location'] = ['2iv','3']
		params['neurotransmitter'] = 'e'
		params['morphology'] = '?'
		# connectivity
		
		params['outputs']['e_cck'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		
	elif cell_type == 'e_grpr':
		params['N'] = 100
		params['location'] = ['1','2o']
		params['neurotransmitter'] = 'e'
		params['morphology'] = '?'
		# connectivity
		
		params['outputs']['pr_pruri'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		
	############################################
	# Inhibitory
	############################################
	
	elif cell_type == 'i_gal':
		params['N'] = 113
		params['location'] = ['1','2o','2id','2iv']
		params['neurotransmitter'] = 'i'
		params['morphology'] = '?'
		# connectivity
		
		params['outputs']['e_vertical'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_grp'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_tac1'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_nts'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_tac2'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		params['outputs']['i_calb2'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		
	elif cell_type == 'i_nnos':
		params['N'] = 77
		params['location'] = ['1','2o','2id','2iv']
		params['neurotransmitter'] = 'i'
		params['morphology'] = '?'
		# connectivity
		
		params['outputs']['pr_noci'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		params['outputs']['i_calb2'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		
	elif cell_type == 'i_npy':
		params['N'] = 150
		params['location'] = ['1','2o','2id','2iv','3']
		params['neurotransmitter'] = 'i'
		params['morphology'] = '?'
		# connectivity
		
		params['outputs']['e_nts'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_tac2'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		params['outputs']['pr_wdr'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		
	elif cell_type == 'i_calb2':
		params['N'] = 113
		params['location'] = ['2o','2id','2iv']
		params['neurotransmitter'] = 'i'
		params['morphology'] = '?'
		# connectivity
		
		params['outputs']['e_grp'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		params['outputs']['i_gal'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		params['outputs']['i_nnos'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
		
	elif cell_type == 'i_pv':
		params['N'] = 226
		params['location'] = ['2iv','3']
		params['neurotransmitter'] = 'i'
		params['morphology'] = '?'
		# connectivity
		
		params['outputs']['e_nts'] = {'p':0.15,
										 'w_mu':1 * nS,
										 'w_sigma': 0.1 * nS,
										 'conductance_name':params['neurotransmitter']}
		params['outputs']['e_tac2'] = {'p':0.15,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}

	return params


def get_afferent_params(cell_type, **kwargs):
		params = {}
		params['outputs'] = {}
		if cell_type == 'pep1':
			params['N'] = 40
			params['delay'] = 100 * ms
			params['location'] = ['1','2o']
			params['neurotransmitter'] = 'e'
			params['stim_electrical_sigma'] = 20 * ms
			params['stim_electrical_rates'] = 5
			
			# connectivity
			
			params['outputs']['pr_noci'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['e_vertical'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['e_grp'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_gal'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_nnos'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_npy'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_calb2'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			
		elif cell_type == 'pep2':
			params['N'] = 12
			params['delay'] = 20 * ms
			params['location'] = ['1','2o']
			params['neurotransmitter'] = 'e'
			params['stim_electrical_sigma'] = 2 * ms
			params['stim_electrical_rates'] = 9
			
			# connectivity
			
			params['outputs']['pr_noci'] = {'p':0.15,
											 'w_mu':2 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
			params['outputs']['e_vertical'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['e_grp'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_gal'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_nnos'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_npy'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			
		elif cell_type == 'trpm8':
			params['N'] = 20
			params['delay'] = 100 * ms
			params['location'] = ['1','2o']
			params['neurotransmitter'] = 'e'
			params['stim_electrical_sigma'] = 20 * ms
			params['stim_electrical_rates'] = 5
			
			# connectivity
			
			params['outputs']['pr_noci'] = {'p':0.15,
											 'w_mu':2 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
			params['outputs']['e_vertical'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			
			params['outputs']['i_gal'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			
		elif cell_type == 'mrgprd':
			params['N'] = 80
			params['delay'] = 100 * ms
			params['location'] = ['2id']
			params['neurotransmitter'] = 'e'
			params['stim_electrical_sigma'] = 20 * ms
			params['stim_electrical_rates'] = 5
			
			# connectivity
			
			params['outputs']['e_vertical'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			
			params['outputs']['i_gal'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			
		elif cell_type == 'np2':
			params['N'] = 10
			params['delay'] = 100 * ms
			params['location'] = ['2id']
			params['neurotransmitter'] = 'e'
			params['stim_electrical_sigma'] = 20 * ms
			params['stim_electrical_rates'] = 5
			
			# connectivity
			
			
		elif cell_type == 'np3':
			params['N'] = 10
			params['delay'] = 100 * ms
			params['location'] = ['2id']
			params['neurotransmitter'] = 'e'
			params['stim_electrical_sigma'] = 20 * ms
			params['stim_electrical_rates'] = 5
			
			# connectivity
			
			
		elif cell_type == 'th':
			params['N'] = 160
			params['delay'] = 100 * ms
			params['location'] = ['2iv']
			params['neurotransmitter'] = 'e'
			params['stim_electrical_sigma'] = 20 * ms
			params['stim_electrical_rates'] = 5
			
			# connectivity
			
			
		elif cell_type == 'AD_ltmr':
			params['N'] = 20
			params['delay'] = 10 * ms
			params['location'] = ['2iv','3','4']
			params['neurotransmitter'] = 'e'
			params['stim_electrical_sigma'] = 2 * ms
			params['stim_electrical_rates'] = 9
			
			# connectivity
			
			params['outputs']['e_nts'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['e_tac2'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_pv'] = {'p':0.15,
												 'w_mu':10 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_gal'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_npy'] = {'p':0.15,
												 'w_mu':2 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			
		elif cell_type == 'AB_ltmr':
			params['N'] = 40
			params['delay'] =0 * ms
			params['location'] = ['2iv','3','4']
			params['neurotransmitter'] = 'e'
			params['stim_electrical_sigma'] = 2 * ms
			params['stim_electrical_rates'] = 9
			
			# connectivity
			
			params['outputs']['e_nts'] = {'p':0.1,
											 'w_mu':1 * nS,
											 'w_sigma': 0.1 * nS,
											 'conductance_name':params['neurotransmitter']}
			params['outputs']['e_tac2'] = {'p':0.1,
												 'w_mu':1 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_pv'] = {'p':0.15,
												 'w_mu':1 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
			params['outputs']['i_gal'] = {'p':0.15,
												 'w_mu':1 * nS,
												 'w_sigma': 0.1 * nS,
												 'conductance_name':params['neurotransmitter']}
		return params


def projection_neuron_types():
	return ['pr_noci','pr_pruri','pr_WDR']

def excitatory_types():
	return ['e_vertical','e_tac1','e_grp','e_nts','e_tac2','e_ckk','e_grpr']

def inhibitory_types():
	return ['i_gal','i_nnos','i_npy','i_calb2','i_pv']

def neuron_types():
	return projection_neuron_types() + excitatory_types() + inhibitory_types()

def c_fibers():
	return ['pep1','mrgprd','trpm8','np2','np3','th']

def a_delta_fibers():
	return ['pep2','AD_ltmr']

def afferent_types():
	return c_fibers() + a_delta_fibers() + ['AB_ltmr']

def intrinsics_table():
	dfs = []
	for nt in neuron_types():
		intrinsics = {}
		intrinsics_ = get_neuron_params(nt)['outputs']
		intrinsics['E_l (mV)'] = intrinsics_['E_l'] * 10 ** 3 
		intrinsics['E_e (mV)'] = intrinsics_['E_e'] * 10 ** 3
		intrinsics['E_i (mV)'] = intrinsics_['E_i'] * 10 ** 3
		intrinsics['C_m (pF)'] = intrinsics_['C_m'] * 10 ** 12
		intrinsics['V_th (mV)'] = intrinsics_['V_th'] * 10 ** 3
		intrinsics['V_r (mV)'] = intrinsics_['V_r'] * 10 ** 3
		intrinsics['tau_e (ms)'] = intrinsics_['tau_e'] * 10 ** 3
		intrinsics['tau_i (ms)'] = intrinsics_['tau_i'] * 10 ** 3
		intrinsics['tau_r (ms)'] = intrinsics_['tau_r'] * 10 ** 3
		intrinsics['g_l_mu (nS)'] = intrinsics_['g_l_mu'] * 10 ** 9
		intrinsics['g_l_sigma (nS)'] = intrinsics_['g_l_sigma'] * 10 ** 9
		intrinsics['I_ex_mu (pA)'] = intrinsics_['I_ex_mu'] * 10 ** 12
		intrinsics['I_ex_sigma (pA)'] = intrinsics_['I_ex_sigma'] * 10 ** 12
		intrinsics['b (pA)'] = intrinsics_['b'] * 10 ** 12
		intrinsics['alpha (nS)'] = intrinsics_['alpha'] * 10 **9
		intrinsics['Delta_T (mV)'] = intrinsics_['Delta_T'] * 10 ** 3
		intrinsics['tau_u (ms)'] = intrinsics_['tau_u'] * 10 ** 3

		dfs.append(pd.DataFrame(intrinsics,index = [nt]))

	return pd.concat(dfs)

def interneuronal_output_param_table(param_name = 'p'):
	
	dfs = []

	nts = neuron_types()
	for pre in nts:
		output_param = {}

		outputs = get_neuron_params(pre)['outputs']
		for post in nts:
			try:
				output_param[post] = outputs[post][param_name]
			except:
				output_param[post] = '?'

		dfs.append(pd.DataFrame(output_param,index = [pre]))
	return pd.concat(dfs)

def interneuronal_connectivity_tables():
	
	p = interneuronal_output_param_table(param_name = 'p')
	w_mu = interneuronal_output_param_table(param_name = 'w_mu')
	w_sigma = interneuronal_output_param_table(param_name = 'w_sigma')


	return p, w_mu, w_sigma
