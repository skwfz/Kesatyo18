# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:16:19 2018

@author: rissans2
"""


from qutip import *
import numpy as np
import matplotlib as m
from matplotlib import pyplot as pl
import matplotlib.colors as colors
import scipy as sc
import util
import tunable_coupling_util as tcu

#%%
levels = 5 #Amount of energy levels for qubits

C = np.array([70e-15,200e-15,70e-15]) #Capasitance of the capacitor in the qubits 1,c,2 These values are different than in the article now!
Cc =  np.array([4e-15,4e-15]) #Capacitance between the coupler and gate 1 and 2
C12 =  np.array(0.1e-15) #Capacitance directly between qubits 1 and 2
omega = np.array([4e9,5e9,4e9])#Frequencies of qubits 1, c and 2

s = tcu.tunable_coupling_simulation(levels,C,Cc,C12,omega)

#%%
omegas_approximate = np.linspace(4.1e9,7e9,2000)
geffs_approximate = np.array([s.gef(o) for o in omegas_approximate])
(omegas_geffint,geffint) = s.gef_accurate_interpolation('/accurate_geffs.npy',newinterpolation=False)
geffs = geffint(omegas_geffint)

#%%
"""Finding out the eigenstates in the decoupled state corresponding to |001>, |100> and their symmetric and antisymmetric combinations."""
eigs = s.sch_Hamiltonian(s.omega[1]).eigenstates()
logical_eigens = s.eigenstates_logical_states(s.omega[1])
Eavg = (logical_eigens[0][0]+logical_eigens[1][0])/2
(symm_real,antisymm_real) = util.corresponding_kets_from_degenerate_subspace(logical_eigens[0][1],logical_eigens[1][1],s.symm_comp,s.antisymm_comp)
(state001_real,state100_real) = util.corresponding_kets_from_degenerate_subspace(logical_eigens[0][1],logical_eigens[1][1],s.state001_comp,s.state100_comp)

#%%
cdict = {
  'red'  :  ( (0.0, 0.25, .25), (1., 1., 1.), (1., 1., 1.)),
  'green':  ( (0.0, 0.0, 0.0), (0.,0.,0.), (0., 0., 0.)),
  'blue' :  ( (0.0, 1.0, 1.0), (1., 1., 1.), (1., 0.45, 0.45)),
}

cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

#%%
pl.close('all')
#%%
"""100 different amplitude square pulses"""
#data_100SP = np.load('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\squarepulse_100amplitudes.npz')
#(TS,OS) = np.meshgrid(data_100SP['times'],s.omega[1]*(1-data_100SP['As']))
#fidelplot = pl.figure()
#pl.pcolor(TS,OS,data_100SP['fidelities'],vmin=0,vmax=1)
#pl.colorbar()
#pl.title("Fidelity of the gate operation with a square pulse")
#pl.xlabel("Time (s)")
#pl.ylabel(r"$\omega_c$")
#
#"""Calculating the theoretical and numerical optimum curves"""
#Ts = np.linspace(0.5e-8,5e-7,1000)
#As = np.zeros(1000)
#Tsaccur = np.linspace(1e-7,5e-7,100)
#Asaccur = np.zeros(100)
#for i in range(0,1000):
#    As[i] = s.theoretical_pulse_amplitude(tcu.square_pulse,Ts[i],s.gef)
#for i in range(0,100):
#    Asaccur[i] = s.theoretical_pulse_amplitude(tcu.square_pulse,Tsaccur[i],geffint)
#Os = abs(s.omega[1]*(1-As))
#Osaccur = abs(s.omega[1]*(1-Asaccur))
#
#"""Some ugly things: finding the optimal omegac fo all times and their indices in the
#region over a "selection curve": the theoretical optimum curve minus some arbitrary constant."""
#best_fidelity_for_time_omegac = np.zeros(1000)
#best_fidelity_for_time = np.zeros(1000)
#for k in range(0,1000):
#    omegac_startindex = abs(s.omega[1]*(1-data_100SP['As'])-(Os[k]-0.15e9)).argmin()#Os and Ts match pretty well with simulation data_200SP['times']
#    best_fidelity_index = data_100SP['fidelities'][omegac_startindex:199,k].argmax()+omegac_startindex
#    best_fidelity_for_time[k] = data_100SP['fidelities'][best_fidelity_index,k]
#    best_fidelity_for_time_omegac[k] = (s.omega[1]*(1-data_100SP['As']))[best_fidelity_index]
#
#pl.plot(Ts,Os,'b')
#pl.plot(Tsaccur,Osaccur,'g')
#pl.plot(Ts,best_fidelity_for_time_omegac)
#pl.plot(Ts,Os-0.15e9)
#pl.legend(["Theoretical best fidelity","Other","Best fidelity above the selection curve","Selection curve"])
#
#phaseplot = pl.figure()
#pl.pcolor(TS,OS,data_100SP['phase_differences'],cmap='seismic', vmin=-np.pi, vmax=np.pi)
#pl.colorbar()
#pl.title("Phase difference to the ideal end state")
#pl.xlabel("Time (s)")
#pl.ylabel(r"$\omega_c$")
#
#pl.figure()
#pl.title("Optimal error as a function of gate time (above the selection curve)")
#pl.plot(Ts,1-best_fidelity_for_time**2)
#pl.xlabel("Time (s)")
#pl.ylabel("Average error")
#pl.yscale("log")

#%%
"""200 different amplitude square pulses, 16 different starting states, average fidelities"""
data_200SP = np.load('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\squarepulse_200amplitudes_16states_avgfidelity.npz')
(TS,OS) = np.meshgrid(data_200SP['times'],s.omega[1]*(1-data_200SP['As']))
pl.figure()
pl.pcolor(TS,OS,np.sqrt(data_200SP['avg_fidelities']),vmin=0)
pl.colorbar()
pl.title("Average fidelity of the gate operation with a square pulse")
pl.xlabel("Time (s)")
pl.ylabel(r"$\omega_c$")

"""Calculating the theoretical and numerical optimum curves"""
Ts = np.linspace(0.5e-8,5e-7,1000)
As = np.zeros(1000)
Tsaccur = np.linspace(1e-7,5e-7,100)
Asaccur = np.zeros(100)
for i in range(0,1000):
    As[i] = s.theoretical_pulse_amplitude(tcu.square_pulse,Ts[i],s.gef)
for i in range(0,100):
    Asaccur[i] = s.theoretical_pulse_amplitude(tcu.square_pulse,Tsaccur[i],geffint)
Os = abs(s.omega[1]*(1-As))#-0.15e9
Osaccur = abs(s.omega[1]*(1-Asaccur))

"""Some ugly things: finding the optimal omegac fo all times and their indices in the
region over a "selection curve": the theoretical optimum curve minus some arbitrary constant."""
best_fidelity_for_time_omegac = np.zeros(1000)
best_fidelity_for_time = np.zeros(1000)
for k in range(0,1000):
    omegac_startindex = abs(s.omega[1]*(1-data_200SP['As'])-(Os[k]-0.15e9)).argmin()#Os and Ts match pretty well with simulation data_200SP['times']
    best_fidelity_index = data_200SP['avg_fidelities'][omegac_startindex:199,k].argmax()+omegac_startindex
    best_fidelity_for_time[k] = data_200SP['avg_fidelities'][best_fidelity_index,k]
    best_fidelity_for_time_omegac[k] = (s.omega[1]*(1-data_200SP['As']))[best_fidelity_index]
    
pl.plot(Ts,Os,'b')
pl.plot(Tsaccur,Osaccur,'g')
pl.plot(Ts,best_fidelity_for_time_omegac)
pl.plot(Ts,Os-0.15e9)
pl.legend(["Theoretical best fidelity","Other","Best fidelity above the selection curve","Selection curve"])

pl.figure()
pl.title("Optimal error as a function of gate time (above the selection curve)")
pl.plot(Ts,1-np.sqrt(best_fidelity_for_time))
pl.xlabel("Time (s)")
pl.ylabel("Average error")
pl.yscale("log")

pl.figure()
pl.pcolor(TS,OS,np.sqrt(1-data_200SP['avg_fidelities']),norm=colors.LogNorm(vmin=(1-data_200SP['avg_fidelities']).min(), vmax=1),cmap='gist_yarg')
pl.colorbar()
pl.title("Average error of the gate operation with a square pulse")
pl.xlabel("Time (s)")
pl.ylabel(r"$\omega_c$")

#%%

"""200 different amplitude square pulses, 16 different starting states, average fidelities, using copmutational basis"""
#data_200SP = np.load('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\squarepulse_200amplitudes_16states_avgfidelity_compstates.npz')
#(TS,OS) = np.meshgrid(data_200SP['times'],s.omega[1]*(1-data_200SP['As']))
#pl.figure()
#pl.pcolor(TS,OS,np.sqrt(data_200SP['avg_fidelities']),vmin=0)
#pl.colorbar()
#pl.title("Average fidelity of the gate operation with a square pulse, computational basis")
#pl.xlabel("Time (s)")
#pl.ylabel(r"$\omega_c$")
#
#"""Calculating the theoretical and numerical optimum curves"""
#Ts = np.linspace(0.5e-8,5e-7,1000)
#As = np.zeros(1000)
#Tsaccur = np.linspace(1e-7,5e-7,100)
#Asaccur = np.zeros(100)
#for i in range(0,1000):
#    As[i] = s.theoretical_pulse_amplitude(tcu.square_pulse,Ts[i],s.gef)
#for i in range(0,100):
#    Asaccur[i] = s.theoretical_pulse_amplitude(tcu.square_pulse,Tsaccur[i],geffint)
#Os = abs(s.omega[1]*(1-As))#-0.15e9
#Osaccur = abs(s.omega[1]*(1-Asaccur))
#
#"""Some ugly things: finding the optimal omegac fo all times and their indices in the
#region over a "selection curve": the theoretical optimum curve minus some arbitrary constant."""
#best_fidelity_for_time_omegac = np.zeros(1000)
#best_fidelity_for_time = np.zeros(1000)
#for k in range(0,1000):
#    omegac_startindex = abs(s.omega[1]*(1-data_200SP['As'])-(Os[k]-0.15e9)).argmin()#Os and Ts match pretty well with simulation data_200SP['times']
#    best_fidelity_index = data_200SP['avg_fidelities'][omegac_startindex:199,k].argmax()+omegac_startindex
#    best_fidelity_for_time[k] = data_200SP['avg_fidelities'][best_fidelity_index,k]
#    best_fidelity_for_time_omegac[k] = (s.omega[1]*(1-data_200SP['As']))[best_fidelity_index]
#    
#pl.plot(Ts,Os,'b')
#pl.plot(Tsaccur,Osaccur,'g')
#pl.plot(Ts,best_fidelity_for_time_omegac)
#pl.plot(Ts,Os-0.15e9)
#pl.legend(["Theoretical best fidelity","Other","Best fidelity above the selection curve","Selection curve"])
#
#pl.figure()
#pl.title("Optimal error as a function of gate time (above the selection curve) (comp basis)")
#pl.plot(Ts,1-np.sqrt(best_fidelity_for_time))
#pl.xlabel("Time (s)")
#pl.ylabel("Average error")


#%%
"""The development of the four eigenstates that take part 
in the swapping process of states |001> and |100>"""
data_FE = np.load('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\swappingprocesseigenstates.npz')

plotstyles = [":","-.","--","-"]
plotcolors = ["b","b","b","b"]

def plot_eigenstate(omegadata,statedata,title,amountofstates,legend,ax2ylim):
    (fig1, ax1) =pl.subplots()
    ax1.set_ylabel("State coefficients")
    ax1.tick_params(axis='y',labelcolor="blue")
    pl.title(title)
    pl.xlabel("Coupler frequency (Hz)")
    for i in range(0,amountofstates-1):#Statedata should be a list of arrays, last element of list is energy
#        ax1.plot(data_FE['omegas'],abs(statedata[i])**2,plotcolors[i]+plotstyles[i],alpha=0.7)
        ax1.plot(omegadata[1:],statedata[i][1:],plotcolors[i]+plotstyles[i],alpha=0.5)#Excluding the degenerate omega[1]-states
    ax1.plot(omegadata,sum([abs(s)**2 for s in statedata[:-1]]))#+abs(statedata[3])**2)
    ax1.legend(legend)
    ax1.set_ylim([-1.05,1.05])
    ax2 = ax1.twinx()
    ax2.set_ylabel("Energy")
    ax2.set_ylim(ax2ylim)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.plot(omegadata,statedata[-1],color='red')

ax2ylim = [2.5e9,5.4e9]

plot_eigenstate(data_FE['omegas'],data_FE['state1'],"1st state development",data_FE['state1'][:,0].size-1,['001','100','010','Sum of norms squared'],ax2ylim)

plot_eigenstate(data_FE['omegas'],data_FE['state2'],"2nd state development",data_FE['state2'][:,0].size-1,['001','100','010','Sum of norms squared'],ax2ylim)

plot_eigenstate(data_FE['omegas'],data_FE['state3'],"3rd state development",data_FE['state3'][:,0].size-1,['001','100','010','Sum of norms squared'],ax2ylim)

#%%
"""The development of the four eigenstates that take part 
in the parasitic cphase-operation"""
data_FE = np.load('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\DefaultParameters\cphaseprocesseigenstates.npz')

ax2ylim = [6e9,11e9]

plot_eigenstate(data_FE['omegas'],data_FE['state1'],"1st state development",data_FE['state1'][:,0].size,['101','110','011','020','Sum of norms squared'],ax2ylim)

plot_eigenstate(data_FE['omegas'],data_FE['state2'],"2nd state development",data_FE['state1'][:,0].size,['101','110','011','020','Sum of norms squared'],ax2ylim)

plot_eigenstate(data_FE['omegas'],data_FE['state3'],"3rd state development",data_FE['state1'][:,0].size,['101','110','011','020','Sum of norms squared'],ax2ylim)

plot_eigenstate(data_FE['omegas'],data_FE['state4'],"4th state development",data_FE['state1'][:,0].size,['101','110','011','020','Sum of norms squared'],ax2ylim)
