# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:52:39 2018

@author: rissans2
"""

from qutip import *
import numpy as np
from matplotlib import pyplot as pl
import scipy as sc
import util
import tunable_coupling_util as tcu
import time
import multiprocessing as mp
import datetime
#%%
"""Simulation constants and the functions directly dependent on them"""
levels = 5 #Amount of energy levels for qubits

C = np.array([70e-15,200e-15,70e-15]) #Capasitance of the capacitor in the qubits 1,c,2 These values are different than in the article now!
Cc =  np.array([4e-15,4e-15]) #Capacitance between the coupler and gate 1 and 2
C12 =  np.array(0.1e-15) #Capacitance directly between qubits 1 and 2
omega = np.array([4e9,5.5e9,4e9])#Frequencies of qubits 1, c and 2. Frequency of c doesn't matter as it's changed automatically later

s = tcu.tunable_coupling_simulation(levels,C,Cc,C12,omega)
#s.omega[1] = 5.2
#s.g = 0.5*(s.Cc/np.sqrt(np.array([s.C[0],s.C[2]])*s.C[1]))*np.sqrt(np.array([s.omega[0],s.omega[2]])*s.omega[1])
#%%
omegas_approximate = np.linspace(4.1e9,7e9,2000)
geffs_approximate = np.array([s.gef(o) for o in omegas_approximate])
(omegas_geffint,geffint) = s.gef_accurate_interpolation('/accurate_geffs.npy',newinterpolation=False)
geffs = geffint(omegas_geffint)

#%%
"""Finding out the eigenstates in the decoupled state corresponding to |001>, |100> and their symmetric and antisymmetric combinations."""
eigs = s.sch_Hamiltonian(s.omega[1]).eigenstates()
logical_eigens = s.eigenstates_logical_states(s.omega[1])#Returns just some combination of |100> and |001> in the degenerate eigenspace
Eavg = (logical_eigens[0][0]+logical_eigens[1][0])/2
(symm_real,antisymm_real) = util.corresponding_kets_from_degenerate_subspace(logical_eigens[0][1],logical_eigens[1][1],s.symm_comp,s.antisymm_comp)
(state001_real,state100_real) = util.corresponding_kets_from_degenerate_subspace(logical_eigens[0][1],logical_eigens[1][1],s.state001_comp,s.state100_comp)
state000_real = util.find_best_ket_match(eigs[1],s.state000_comp)
state101_real = util.find_best_ket_match(eigs[1],s.state101_comp)
state020_real = util.find_best_ket_match(eigs[1],s.state020_comp)#Note: not all of these match perfectly
state011_110_sym_real = util.find_best_ket_match(eigs[1],(s.state011_comp+s.state110_comp).unit())
state011_110_antisym_real = util.find_best_ket_match(eigs[1],(s.state011_comp-s.state110_comp).unit())


#%%
"""The simulation with 100 different square_pulse amplitudes and 1000 different times using a single starting state. """
#S0 = state100_real
#Scompare = state001_real*(1.0j)
#"""The simulation parameters given to the solver"""
#used_pulse = tcu.square_pulse
#T = 5e-7
#H = s.simulation_Hamiltonian(used_pulse=used_pulse)
#N = 100 #For how many different amplitudes the simulation is run
#fidelities = np.zeros((N,1000))
#phase_differences = np.zeros((N,1000))
#times = np.linspace(0,T,1000)
#As = np.linspace((s.omega[1]-s.omega[0])/s.omega[1],0,N)
#for k in range(0,N):
#    #A = s.theoretical_pulse_amplitude(used_pulse = used_pulse, T=T, used_gef = geffint)
#    A = As[k]
#    """Running the simulation"""
#    args = {'T':T,'A':A}
#    S0 = state100_real
#    opts = solver.Options(num_cpus=6, atol= 1e-11, rtol= 1e-9,store_states=True, nsteps=1000)
#    res = mesolve(H,S0,times,[],[],options=opts,args = args)
#    fidelities[k] = np.array([abs((Scompare.dag()*state).data.toarray()[0][0]) for state in res.states])
#    phase_differences[k] = np.array([util.angle_between_states(Scompare,state) for state in res.states])
#np.savez('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\squarepulse_100amplitudes.npz',times=times,As=As,fidelities=fidelities,phase_differences=phase_differences)
#%%
"""200 different square pulse amplitudes, 16 different starting states for each."""
ket0 = basis(s.levels,0)
ket1 = basis(s.levels,1)
ketx = (basis(s.levels,0)+basis(s.levels,1)).unit()
kety = (basis(s.levels,0)+1.0j*basis(s.levels,1)).unit()
kets = [ket0,ket1,ketx,kety]

used_pulse = tcu.square_pulse

T = 5e-7
H = s.simulation_Hamiltonian(used_pulse=used_pulse)
N = 200 #For how many different amplitudes the simulation is run
timesteps = 1000
avg_fidelities = np.zeros((N,timesteps))
phis = np.zeros((N,timesteps))
all_fidelities = [np.zeros((16,timesteps)) for i in range(0,N)]
times = np.linspace(0,T,timesteps)
As = np.linspace(2*(s.omega[1]-s.omega[0])/s.omega[1],0,N)
starting_states_coefs = util.g_e_x_y_coefficients()

start = time.time()#For measuring the time it took to calculate everything

#Finding the correct phis to rotate the qubits, using just some state
#for k in range(0,N):
#    A = As[k]
#    opts = solver.Options(num_cpus=6, atol= 1e-11, rtol= 1e-9,store_states=True, nsteps=1500)
#    args = {'T':T,'A':A}
#    S0 = state000_real+state001_real+state100_real+state101_real
#    Scompare = state000_real+(1.0j)*state001_real+(1.0j)*state100_real+state101_real
#    res = mesolve(H,S0,times,[],[],options=opts,args = args)
#    for i in range(timesteps):
#        fixres = s.dynamic_phase_fixer_5(res.states[i],Scompare,state000_real,state101_real)
#        phis[k,i] = fixres[1]
#    print("Phis calculated for", k, ". amplitude: ", A)
Scompares = []
for coefs in starting_states_coefs:
    Scompares.append(s.state000_real*coefs[0]+1.0j*s.state001_real*coefs[2]\
            +1.0j*s.state100_real*coefs[1]+s.state101_real*coefs[3])

for k in range(0,N):
    A = As[k]#to be replaced
    opts = solver.Options(num_cpus=6, atol= 1e-11, rtol= 1e-9,store_states=True, nsteps=1500)
    args = {'T':T,'A':A}
    
    #The arrays of state vectors for all different starting states on this amplitude
    amplitude_states = np.zeros(16,timesteps,object)
    #Fidelities for this all the different starting states on this amplitude
    amplitude_fidelities = np.zeros(16,timesteps)
    #Phis for all all different times on this amplitude
    amplitude_phis = np.zeros(timesteps)
    #The arrays of fixed state vectors for all different starting states on this amplitude
    amplitude_fixed_states = np.zeros(16,timesteps,object)
    
    #Doing the simulation
    for coefs in starting_states_coefs:
        amplitude_states.append(s.simulate_coefs(coefs,times,H,opts,args))
        print("Coefficients ",coefs, " calculated. Elapsed time:",str(datetime.timedelta(seconds=time.time()-start)))
    
    #Doing the individual qubit rotation optimization and calculating results
    for i in range(timesteps):
        optimres = s.dynamic_phase_fixer_6(amplitude_states[:,i],Scompares)
        amplitude_phis[i] = optimres[1]
        avg_fidelities[k,i] = optimres[0]
        amplitude_fixed_states[:,i] = np.array(optimres[2])
        all_fidelities[k][:,i] = np.array([abs((Scompare.dag()*state).data.toarray()[0][0])**2 for (Scompare,state) in zip(Scompares,amplitude_fixed_states[:,i])])
    
    print("Fidelities calculated for ", k, ". amplitude: ", A)

end = time.time()

print("Amount of time elapsed: ",  str(datetime.timedelta(seconds=end-start)))   

np.savez('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\squarepulse_200amplitudes_16states_avgfidelity.npz'\
         ,times=times,As=As,avg_fidelities=avg_fidelities,all_fidelities=all_fidelities)
#%%

"""200 different square pulse amplitudes, 16 different starting states for each, 
but using the simpler, computational basis."""
#ket0 = basis(s.levels,0)
#ket1 = basis(s.levels,1)
#ketx = (basis(s.levels,0)+basis(s.levels,1)).unit()
#kety = (basis(s.levels,0)+1.0j*basis(s.levels,1)).unit()
#kets = [ket0,ket1,ketx,kety]
#
#used_pulse = tcu.square_pulse
#
#T = 5e-7
#H = s.simulation_Hamiltonian(used_pulse=used_pulse)
#N = 200 #For how many different amplitudes the simulation is run
#avg_fidelities = np.zeros((N,1000))
#all_fidelities = [np.zeros((16,1000)) for i in range(0,N)]
#all_phase_differences = [np.zeros((16,1000)) for i in range(0,N)]
#avg_phase_differences = np.zeros((N,1000))
#times = np.linspace(0,T,1000)
#As = np.linspace(2*(s.omega[1]-s.omega[0])/s.omega[1],0,N)
#
#start = time.time()#For measuring the time it took to calculate everything
#
#for k in range(0,N):
#
#    A = As[k]#to be replaced
#    opts = solver.Options(num_cpus=6, atol= 1e-11, rtol= 1e-9,store_states=True, nsteps=1000)
#    args = {'T':T,'A':A}
#    
#    #Fidelities and phase differences for this all the different states on this amplitude
#    amplitude_fidelities = []
#    amplitude_phase_differences = []
#    
#    for k1 in kets:
#        for k2 in kets:
#            S0 = tensor(k1,basis(s.levels,0),k2)
#            Scompare = S0[0][0][0]*s.state000_comp+1.0j*S0[1][0][0]*s.state100_comp\
#                +1.0j*S0[levels**2][0][0]*s.state001_comp+S0[levels**2+1][0][0]*s.state101_comp#iswapped state
#            
#            res = mesolve(H,S0,times,[],[],options=opts,args = args)
#            
#            amplitude_fidelities.append(np.array([abs((Scompare.dag()*state).data.toarray()[0][0])**2 for state in res.states]))
#            amplitude_phase_differences.append(np.array([util.angle_between_states(Scompare,state) for state in res.states]))
#            print("Coefficients calculated for one starting state of amplitude ", A)
#    for i in range(0,16):
#        all_fidelities[k][i] = amplitude_fidelities[i]
#        all_phase_differences[k][i] = amplitude_fidelities[i]
#    avg_fidelities[k] = np.array([np.mean(fs) for fs in zip(*amplitude_fidelities)])
#    avg_phase_differences[k] = np.array([np.mean(pds) for pds in zip(*amplitude_phase_differences)])
#    print("Fidelities calculated for ", k, ". amplitude: ", A)
#
#end = time.time()
#
#print("Amount of time elapsed: ",  str(datetime.timedelta(seconds=end-start)))   
#
#np.savez('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\squarepulse_200amplitudes_16states_avgfidelity_compstates.npz'\
#         ,times=times,As=As,avg_fidelities=avg_fidelities,avg_phase_differences=avg_phase_differences\
#         ,all_fidelities=all_fidelities,all_phase_differences=all_phase_differences)
#%%
"""Just a normal simulation for plotting states below."""
S0 = ((1)*s.state001_real+(1)*s.state100_real+(1)*s.state101_real+(1)*s.state000_real).unit()
Scompare = ((1)*s.state101_real+(1)*s.state000_real+(1.0j)*s.state100_real+(1.0j)*s.state001_real).unit()
used_pulse = tcu.square_pulse
T = 5e-7
H = s.simulation_Hamiltonian(used_pulse=used_pulse)
times = np.linspace(0,T,1000)
A = 0.4#s.theoretical_pulse_amplitude(used_pulse,T,geffint)#0.19627
args = {"A":A, "T":T}
start = time.time()
opts = solver.Options(num_cpus=6, atol= 1e-11, rtol= 1e-9,store_states=True, nsteps=1000)
res_onesimulation = mesolve(H,S0,times,[],[],options=opts,args = args)
print(time.time()-start)

res_onesimulation_fidelities = np.zeros(1000)
res_onesimulation_phis = np.zeros(1000)

start = time.time()
for i in range(1000):
    fixres = s.dynamic_phase_fixer_5(res_onesimulation.states[i],Scompare,s.state000_real,s.state101_real)
    res_onesimulation_phis[i] = fixres[1]
    res_onesimulation.states[i] = fixres[2]
    res_onesimulation_fidelities[i] = fixres[0]
end = time.time()
print(end-start)

res_onesimulation_fidelities = np.array([abs((Scompare.dag()*state).data.toarray()[0][0])**2 for state in res_onesimulation.states])



#%%
"""Calculates the development of the three/four eigenstates that take part 
in the swapping process of states |001> and |100>. A fourth state doesn't seem to be found (would correspond to  |111> state at omega[1])"""
#N = 500
#omegas = np.linspace(s.omega[1],2*s.omega[0]-s.omega[1],N)
#(state1,state2,state3,state4) = (np.zeros((5,N)),np.zeros((5,N)),np.zeros((5,N)),np.zeros((5,N)))
#previous_swappingstates = [[0,s.state100_comp+s.state001_comp],[0,s.state100_comp-s.state001_comp],[0,s.state010_comp]]
#for k in range(0,N):
#    omegac = omegas[k]
#    swappingstates = s.find_swapping_eigenstates(omegac)
#    swappingstates[0] = (swappingstates[0][0],util.correct_corresponding_ket_sign(previous_swappingstates[0][1],swappingstates[0][1]))
#    swappingstates[1] = (swappingstates[1][0],util.correct_corresponding_ket_sign(previous_swappingstates[1][1],swappingstates[1][1]))
#    swappingstates[2] = (swappingstates[2][0],util.correct_corresponding_ket_sign(previous_swappingstates[2][1],swappingstates[2][1]))
#    state1[0][k] = (s.state001_comp.dag()*swappingstates[0][1])[0][0][0]
#    state1[1][k] = (s.state100_comp.dag()*swappingstates[0][1])[0][0][0]
#    state1[2][k] = (s.state010_comp.dag()*swappingstates[0][1])[0][0][0]
#    state1[3][k] = (s.state111_comp.dag()*swappingstates[0][1])[0][0][0]
#    state1[4][k] = swappingstates[0][0]
#    state2[0][k] = (s.state001_comp.dag()*swappingstates[1][1])[0][0][0]
#    state2[1][k] = (s.state100_comp.dag()*swappingstates[1][1])[0][0][0]
#    state2[2][k] = (s.state010_comp.dag()*swappingstates[1][1])[0][0][0]
#    state2[3][k] = (s.state111_comp.dag()*swappingstates[1][1])[0][0][0]
#    state2[4][k] = swappingstates[1][0]
#    state3[0][k] = (s.state001_comp.dag()*swappingstates[2][1])[0][0][0]
#    state3[1][k] = (s.state100_comp.dag()*swappingstates[2][1])[0][0][0]
#    state3[2][k] = (s.state010_comp.dag()*swappingstates[2][1])[0][0][0]
#    state3[3][k] = (s.state111_comp.dag()*swappingstates[2][1])[0][0][0]
#    state3[4][k] = swappingstates[2][0]
#    #state4[0][k] = (s.state001_comp.dag()*swappingstates[3][1])[0][0][0]
#    #state4[1][k] = (s.state100_comp.dag()*swappingstates[3][1])[0][0][0]
#    #state4[2][k] = (s.state010_comp.dag()*swappingstates[3][1])[0][0][0]
#    #state4[3][k] = (s.state111_comp.dag()*swappingstates[3][1])[0][0][0]
#    previous_swappingstates = swappingstates
#    if(k%10 == 0):
#        print(k)
#np.savez('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\swappingprocesseigenstates.npz',omegas=omegas,state1=state1,state2=state2,state3=state3,state4=state4)

#%%
"""Calculates the deleopment of the four eigenstates that affect the development of the 101 state. Doesn't calculate in the < omega[0] region"""
#N = 500
#omegas = np.linspace(s.omega[1],s.omega[0],N)
#(state1,state2,state3,state4) = (np.zeros((5,N)),np.zeros((5,N)),np.zeros((5,N)),np.zeros((5,N)))
#previous_swappingstates = [[0,s.state101_comp],[0,s.state011_comp+s.state110_comp],[0,s.state011_comp-s.state110_comp],[0,s.state020_comp]]
#
#for k in range(0,N):
#    omegac = omegas[k]
#    swappingstates = s.find_cphase_eigenstates(omegac)
#    swappingstates[0] = (swappingstates[0][0],util.correct_corresponding_ket_sign(previous_swappingstates[0][1],swappingstates[0][1]))
#    swappingstates[1] = (swappingstates[1][0],util.correct_corresponding_ket_sign(previous_swappingstates[1][1],swappingstates[1][1]))
#    swappingstates[2] = (swappingstates[2][0],util.correct_corresponding_ket_sign(previous_swappingstates[2][1],swappingstates[2][1]))
#    swappingstates[3] = (swappingstates[3][0],util.correct_corresponding_ket_sign(previous_swappingstates[3][1],swappingstates[3][1]))
#    state1[0][k] = (s.state101_comp.dag()*swappingstates[0][1])[0][0][0]
#    state1[1][k] = (s.state011_comp.dag()*swappingstates[0][1])[0][0][0]
#    state1[2][k] = (s.state110_comp.dag()*swappingstates[0][1])[0][0][0]
#    state1[3][k] = (s.state020_comp.dag()*swappingstates[0][1])[0][0][0]
#    state1[4][k] = swappingstates[0][0]
#    state2[0][k] = (s.state101_comp.dag()*swappingstates[1][1])[0][0][0]
#    state2[1][k] = (s.state011_comp.dag()*swappingstates[1][1])[0][0][0]
#    state2[2][k] = (s.state110_comp.dag()*swappingstates[1][1])[0][0][0]
#    state2[3][k] = (s.state020_comp.dag()*swappingstates[1][1])[0][0][0]
#    state2[4][k] = swappingstates[1][0]
#    state3[0][k] = (s.state101_comp.dag()*swappingstates[2][1])[0][0][0]
#    state3[1][k] = (s.state011_comp.dag()*swappingstates[2][1])[0][0][0]
#    state3[2][k] = (s.state110_comp.dag()*swappingstates[2][1])[0][0][0]
#    state3[3][k] = (s.state020_comp.dag()*swappingstates[2][1])[0][0][0]
#    state3[4][k] = swappingstates[2][0]
#    state4[0][k] = (s.state101_comp.dag()*swappingstates[3][1])[0][0][0]
#    state4[1][k] = (s.state011_comp.dag()*swappingstates[3][1])[0][0][0]
#    state4[2][k] = (s.state110_comp.dag()*swappingstates[3][1])[0][0][0]
#    state4[3][k] = (s.state020_comp.dag()*swappingstates[3][1])[0][0][0]
#    state4[4][k] = swappingstates[3][0]
#    previous_swappingstates = swappingstates
#    if(k%10 == 0):
#        print(k)
#
#np.savez('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\cphaseprocesseigenstates.npz',omegas=omegas,state1=state1,state2=state2,state3=state3,state4=state4)

#%%
"""Finds an optimal sine pulse for a certain amplitude"""

#T = 5e-8
#S0 = state100_real
#Scompare = state001_real*(1.0j)
#
#used_pulse = tcu.sine_pulse
#H = s.simulation_Hamiltonian(used_pulse=used_pulse)
#
#opts = solver.Options(num_cpus=6, atol= 1e-11, rtol= 1e-9,store_states=True, nsteps=1000)
#
#def errors_of_sine_pulse(Aarray):
#    errors = np.zeros(Aarray.size)
#    for k in range(0,Aarray.size):
#        A = Aarray[k]#Tarray[k]
#        args = {"A":A, "T":T}
#        times = np.linspace(0,T,1000)
#        res = mesolve(H,S0,times,[],[],options=opts,args = args)
#        error = 1-abs((Scompare.dag()*res.states[-1])[0][0][0])**2
#        errors[k] = error
#    return errors
#
##def errors_of_sine_pulse(T):
##    T
##    args = {"A":A, "T":T}
##    times = np.linspace(0,T,1000)
##    res = mesolve(H,S0,times,[],[],options=opts,args = args)
##    error = 1-abs((Scompare.dag()*res.states[-1])[0][0][0])**2
##    return error
#
##bounds = sc.optimize.Bounds(0.2, 0.3, keep_feasible=False)
#optimize_res = sc.optimize.minimize(errors_of_sine_pulse,x0=0.3,bounds=[(0.2,0.4)])
#
##optimize_res = sc.optimize.brute(errors_of_sine_pulse,ranges=(slice(0.24,0.3),))
#
#args = {"A":optimize_res.x,"T":T}
#times = np.linspace(0,T,1000)
#res_onesimulation = mesolve(H,S0,times,[],[],options=opts,args = args)

#%%
"""Plotting"""
pl.close('all')

"""Plotting"""
#|001>-state
fig1 = tcu.plot_state_evolution("State "+r"|001$\rangle$"+" superposition coefficient",times,res_onesimulation.states,s.state001_comp)

#|100>-state
fig2 = tcu.plot_state_evolution("State "+r"|100$\rangle$"+" superposition coefficient",times,res_onesimulation.states,s.state100_comp)

#|010>-state. Qubits 1 and 2 should exchange energy here.
fig3 = tcu.plot_state_evolution("State "+r"|010$\rangle$"+" superposition coefficient",times,res_onesimulation.states,s.state010_comp)

#|111>-state. Qubits 1 and 2 should exchange energy here.
fig3 = tcu.plot_state_evolution("State "+r"|111$\rangle$"+" superposition coefficient",times,res_onesimulation.states,s.state111_comp)

#Calculated eigenstate in Schrödinger picture corresponding to |100>
fig4 = tcu.plot_state_evolution("The real state "+r"|100$\rangle$"+" superposition coefficient",times,res_onesimulation.states,state100_real)

#Calculated eigenstate in Schrödinger picture corresponding to |001>
fig4 = tcu.plot_state_evolution("The real state "+r"|001$\rangle$"+" superposition coefficient",times,res_onesimulation.states,state001_real)

#Calculated eigenstate in Schrödinger picture corresponding to |001>+|100>
fig4 = tcu.plot_state_evolution("The real state "+r"|001$\rangle$+|100$\rangle$"+" superposition coefficient",times,res_onesimulation.states,symm_real)

#Calculated eigenstate in Schrödinger picture corresponding to |001>-|100>
fig4 = tcu.plot_state_evolution("The real state "+r"|001$\rangle$-|100$\rangle$"+" superposition coefficient",times,res_onesimulation.states,antisymm_real)

fig5 = tcu.plot_state_evolution("The real state "+r"|000$\rangle$"+" superposition coefficient",times,res_onesimulation.states,state000_real)

fig6 = tcu.plot_state_evolution("The real state "+r"|101$\rangle$"+" superposition coefficient",times,res_onesimulation.states,state101_real)

fig7 = pl.figure()
pl.plot(times,res_onesimulation_fidelities)

#fig5 = tcu.plot_state_evolution("The operation state corresponding to "+r"|001$\rangle$"+" superposition coefficient",times,res.states,state001_operation)

#fig6 = tcu.plot_state_evolution("The operation state corresponding to "+r"|100$\rangle$"+" superposition coefficient",times,res.states,state100_operation)

#geff: theoretical and numerically calculated

figgef = pl.figure()
pl.title(r"$\tilde g$ as a function of $\omega_c$")
pl.plot(omegas_approximate,geffs_approximate)
pl.plot(omegas_geffint,geffs,'o',markersize=2)
pl.plot(omegas_geffint,geffint(omegas_geffint))
pl.legend(["Theoretical ","Numerically calculated values","Spline fit"])
pl.xlabel(r"$\omega_c$")