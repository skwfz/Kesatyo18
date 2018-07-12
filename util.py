# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:50:18 2018

@author: rissans2
"""
from qutip import *
import numpy as np
from matplotlib import pyplot as pl
import scipy as sc


def kets_column_stack(qlist,levels):
    """Makes a matrix out of the given array of ket-vectors by stacking them side by side in the
    order given in qlist. Works correctly only for combined states of 3 states each with 'levels'
    amount of eigenstates."""
    vecs = [a.data.toarray() for a in qlist]#Change from Qobj to numpy array
    eigMatrix = Qobj(np.column_stack(vecs))
    eigMatrix.dims = [[levels, levels, levels], [levels, levels, levels]]
    return eigMatrix

def find_best_ket_match(qlist,ket):
    """Returns the vector from qlist which inner product with 'ket' is largest"""
    maxvals = [(abs((q.dag()*ket)[0]),i) for (q,i) in zip(qlist,range(qlist.size))]#List of (array[[inner product]],index)
    maxindex = max([(a[0][0],i) for (a,i) in maxvals], key = lambda t: t[0])[1]
    if((qlist[maxindex].dag()*ket)[0][0] < 0):
        return -qlist[maxindex]
    else:
        return qlist[maxindex]

def find_best_ket_match_index(qlist,ket):
    """Returns the vector from qlist which inner product with 'ket' is largest"""
    maxvals = [(abs((q.dag()*ket)[0]),i) for (q,i) in zip(qlist,range(qlist.size))]#List of (array[[inner product]],index)
    maxindex = max([(a[0][0],i) for (a,i) in maxvals], key = lambda t: t[0])[1]
    return maxindex

def normal_kets_from_best_ket_match(qlist,ket1,ket2):
    """If find_best_ket_match finds the kets corresponding to ket1 and ket2 from qlist, it returns them.
    Otherwise the function assumes that it can find the kets corresponding to their symmetric and antisymmetric combinations,
    and returns the kets that match ket1 and ket2 by combining the symmetric and antisymmetric vectors."""
    """TODO: possibly doesn't work if vectors have complex coefficients"""
    k1 = find_best_ket_match(qlist,ket1)
    k2 = find_best_ket_match(qlist,ket2)
    if(abs(k1.dag()*ket1) > 0.85):#Arbitrary cutoff
        if(np.real((k1.dag()*ket1).data.toarray()[0][0]) <= -0.85):
            k1 = -k1
        if(np.real((k2.dag()*ket2).data.toarray()[0][0])) <= -0.85:
            k2 = -k2
        return (k1,k2)
    else:
        symm = find_best_ket_match(qlist,(ket1+ket2).unit())
        antisymm = find_best_ket_match(qlist,(ket1-ket2).unit())
        (ketcand1, ketcand2) = ((symm+antisymm).unit(), (symm-antisymm).unit())
        if(abs(ketcand1.dag()*ket1) < 0.85):
            (ketcand1,ketcand2) = (ketcand2,ketcand1)
        if(np.real((ketcand1.dag()*ket1).data.toarray()[0][0]) <= -0.85):#Wrong sign
            ketcand1 = -ketcand1
        if(np.real((ketcand2.dag()*ket2).data.toarray()[0][0]) <= -0.85):
            ketcand2 = -ketcand2
        return (ketcand1, ketcand2)

def kets_from_best_ket_match(qlist,ket1,ket2):
    """Works similarly as normal_kets_from_best_ket_match but if ket1 and ket2 don't correspond to any vectors in qlist,
    it is assumed that their symmetric and antisymmetric combinations do and they are returned."""
    k1 = find_best_ket_match(qlist,ket1)
    k2 = find_best_ket_match(qlist,ket2)
    if(abs(k1.dag()*ket1) > 0.85):#Arbitrary cutoff
        if(0 >= np.real((k1.dag()*ket1).data.toarray()[0][0])):
            k1 = -k1
        if(0 >= np.real((k2.dag()*ket2).data.toarray()[0][0])):
            k2 = -k2
        return (k1,k2)
    else:
        symm = find_best_ket_match(qlist,(ket1+ket2).unit())
        antisymm = find_best_ket_match(qlist,(ket1-ket2).unit())
        if(0 >= np.real((symm.dag()*(ket1+ket2)).data.toarray()[0][0])):#If the sign is wrong. 
            symm = -symm
        if(0 >= np.real((antisymm.dag()*(ket1-ket2)).data.toarray()[0][0])):#If the sign is wrong
            antisymm = -antisymm
        return (symm,antisymm)

def correct_corresponding_ket_sign(ketc,ket):
    """If ket is supposed to match with ket, returns ket or -ket depending which sign matches
    the sign of ketc"""
    if(0 >= np.real((ketc.dag()*ket).data.toarray()[0][0])):
        return -ket
    else:
        return ket

def corresponding_kets_from_degenerate_subspace(ket1,ket2,corr1,corr2):
    """Finds the combinations of vectors ket1 and ket2 so that the combinations match the vectors corr1 and corr2 
    as well as possible. It is assumed that ket1 and ket2 are orthogonal and the combinations of the vectors must be
    orthogonal as well. Also assumed that ket1,ket2,corr1,corr2 don't have complex coefficients.
    Accounts for the fact that ket1 and ket2 have a 'handedness', their relative signs matter."""
    def uncorrespondingness(theta,ket1,ket2):
        #matching = corr1.dag()*(a*ket1+np.sqrt(1-a**2)*ket2) + corr2.dag()*(np.sqrt(1-a**2)*ket1+a*ket2)#When inner product of both is 1, matching is perfect
        matching = corr1.dag()*(np.cos(theta)*ket1+np.sin(theta)*ket2) + corr2.dag()*(np.cos(theta+np.pi/2)*ket1+np.sin(theta+np.pi/2)*ket2)
        return -np.real((matching)[0][0][0])
    theta1 = sc.optimize.minimize_scalar(uncorrespondingness,bounds=[0,2*np.pi],method='bounded',args=(ket1,ket2)).x
    theta2 = sc.optimize.minimize_scalar(uncorrespondingness,bounds=[0,2*np.pi],method='bounded',args=(ket1,-ket2)).x
    if(uncorrespondingness(theta1,ket1,ket2) < uncorrespondingness(theta2,ket1,-ket2)):
        (res1,res2) = (np.cos(theta1)*ket1+np.sin(theta1)*ket2,np.cos(theta1+np.pi/2)*ket1+np.sin(theta1+np.pi/2)*ket2)
    else:
        (res1,res2) = (np.cos(theta2)*ket1+np.sin(theta2)*(-ket2),np.cos(theta2+np.pi/2)*ket1+np.sin(theta2+np.pi/2)*(-ket2))
    return (res1,res2)

def angle_between_states(ket1,ket2):
    """If ket1 and ket2 match apart from a phase difference, returns the complex angle that ket2 is rotated 
    from ket1."""    
    #solution = sc.optimize.minimize_scalar(lambda theta: np.angle(1-(ket1.dag()*ket2*np.exp(1.0j*theta))[0][0][0]),bounds=[-np.pi,np.pi],method='bounded')
    #return solution.x
    return np.angle((ket1.dag()*ket2)[0][0][0])

def sigmax_large(levels):
    """Returns the sigmax-matrix, but it's size is levels x levels."""
    return basis(levels,1)*basis(levels,0).dag() + basis(levels,0)*basis(levels,1).dag()

def sigmay_large(levels):
    """Returns the sigmay-matrix, but it's size is levels x levels."""
    return 1.0j*basis(levels,1)*basis(levels,0).dag() -1.0j*basis(levels,0)*basis(levels,1).dag()

def sigmaz_large(levels):
    """Returns the sigmaz-matrix, but it's size is levels x levels."""
    return basis(levels,0)*basis(levels,0).dag() - basis(levels,1)*basis(levels,1).dag()

def iSWAP_large(levels):
    """Returns the iSWAP-matrix, but it's size is levels x levels."""
    return (tensor(sigmax_large(levels),sigmax_large(levels))+tensor(sigmay_large(levels),sigmay_large(levels)))*1.0j/2 \
            + (tensor(sigmaz_large(levels),sigmaz_large(levels))+tensor(qeye(levels),qeye(levels)))*1/2

def iSWAP_three(levels):
    """Returns the iSWAP-matrix, but for a Hilbert space of three qubits with levels amount of energy 
    levels each."""
    return (tensor(sigmax_large(levels),qeye(levels),sigmax_large(levels))+tensor(sigmay_large(levels),qeye(levels),sigmay_large(levels)))*1.0j/2 \
            + (tensor(sigmaz_large(levels),qeye(levels),sigmaz_large(levels))+tensor(qeye(levels),qeye(levels),qeye(levels)))*1/2

def g_e_x_y_coefficients():
    """Used for finding the 16 linearly independent basis states for the combined Hilbert space
    of two qubits (or really the states corresponding to the 16 linearly independent density
    matrices). Works as follows: |0>,|1>,|0>+|1>,|0>+i|1> are the basis states for one qubit. This
    function returns the |00>,|01>,|10>,|11> coefficients of all possible tensor products of those
    basis states. (in the order: |00>,|01>,|10>,|11>) The name refers to ground,excited,sigmax-eigenvector,sigmay-eigenvector"""
    n = 1/np.sqrt(2)
    h = 1/2
    return ((1,0,0,0),(0,1,0,0),(n,n,0,0),(n,1.0j*n,0,0),#|0> x (|0>,|1>,|0>+|1>,|0>+i|1>)
            (0,0,1,0),(0,0,0,1),(0,0,n,n),(0,0,n,1.0j*n),#|1> x (|0>,|1>,|0>+|1>,|0>+i|1>)
            (n,0,n,0),(0,n,0,n),(h,h,h,h),(h,1.0j*h,h,1.0j*h),#|0>+|1> x (|0>,|1>,|0>+|1>,|0>+i|1>)
            (n,0,1.0j*n,0),(0,n,0,1.0j*n),(h,h,1.0j*h,1.0j*h),(h,1.0j*h,1.0j*h,-h))#|0>+i|1> x (|0>,|1>,|0>+|1>,|0>+i|1>)

def positive_angle(num):
    angle = np.angle(num)
    if angle > 0:
        return angle
    else:
        return np.pi-angle