#!/usr/bin/env python
# coding: utf-8

# In[1]:
import matplotlib.pyplot as plt
import numpy as np
from qutip import *

class System:
    def __init__(self, N = 4, M = 0):
        #make operator lists for N and M
        self.N = N
        self.M = M
        
        si_list = []
        sx_list = []
        sy_list = []
        sz_list = []
        sp_list = []
        sm_list = []
        
        si = qeye(2)
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()
        sm = destroy(2)
        sp = create(2)
        
        if(M>0): bi = qeye(M)
            
        for n in range(N):
            op_list = []
            for m in range(N):
                op_list.append(si)
          
            if(M>0): op_list.append(bi)   

            op_list[n] = sx
            sx_list.append(tensor(op_list))

            op_list[n] = sy
            sy_list.append(tensor(op_list))

            op_list[n] = sz
            sz_list.append(tensor(op_list))

            op_list[n] = sp
            sp_list.append(tensor(op_list))

            op_list[n] = sm
            sm_list.append(tensor(op_list))
            
            op_list[n] = si
            
        self.op_list = op_list
        self.si_list = si_list
        self.sx_list = sx_list
        self.sy_list = sy_list
        self.sz_list = sz_list
        self.sp_list = sp_list
        self.sm_list = sm_list
        
        self.c_ops = []    
        self.r_ops = []
        
        # make default collapse op list (empty) and return op list (sigmaz for each site)
        for n in range(self.N):
            self.r_ops.append(sz_list[n])
       
        if(self.M>0):
            aop = destroy(M)
            op_list[N] = aop
            a = tensor(op_list)
            self.a = a
            a = self.a
            self.r_ops.append(a.dag()*a)
        
        #default parameters
        
        self.n_th = 0.0 #for thermal bath
        self.w0 = 1.0*2*np.pi #atom frequency
        self.omegac = self.w0 #cavity frequency
        self.gamma = 0.0 # atom emission (dissipation)
        self.pump = 0.0
        self.Jx = .1  *np.ones(self.N) #spin interaction terms
        self.Jy = .1 *np.ones(self.N)
        self.Jz = 0.0*np.ones(self.N)
        self.H = 0
        self.psi0 = 0
        self.g = -1
        self.kappa = 0
        
        Dicke = False
    
    def setJconst(self, jx,jy,jz): #input is a single int or float (interaction terms are equal for each site)
        self.Jx = jx *np.ones(self.N)
        self.Jy = jy  *np.ones(self.N)
        self.Jz = jz  *np.ones(self.N)
    def setFreq(self, w0 = 0, omegac = 0):
        if w0 !=0 : self.w0 = w0
        if omegac != 0: self.omegac = omegac
        return self.w0, self.omegac
    def SpinCOps(self,gamma = 0, pump = 0): 
        #make collapse operators for spin sites
        c_ops = self.c_ops
        sz_list = self.sz_list
        sm_list = self.sm_list
        sp_list = self.sp_list
        N = self.N
        
        if(isinstance(gamma, (float, int))): #input is one value
            for n in range(N):
                if(gamma>0.0):
                    c_ops.append(np.sqrt(gamma)*sm_list[n])

        elif(len(gamma)==len(sz_list)):#input is list of values
            for n in range(N):
                if(gamma[n]>0.0):
                    c_ops.append(np.sqrt(gamma[n])*sm_list[n])
                   # print("nth gamma c_op appended")
        else:
            return "invalid input"
        
        if(isinstance(pump, (float, int))): #input is one value
            for n in range(N):
                if(pump>0.0):
                    c_ops.append(np.sqrt(pump)*sp_list[n])

        elif(len(pump)==len(sz_list)):#input is list of values
            for n in range(N):
                if(pump[n]>0.0):
                    c_ops.append(np.sqrt(pump[n])*sp_list[n])
                    #print("nth pump c_op appended")
        else:
            return "invalid input"
        self.c_ops = c_ops
        #print("len of c_ops is : ",len(self.c_ops))
        self.gamma = gamma
        self.pump = pump
    def CavCOps(self, kappa= 0, n_th= 0.0):
        if(self.M==0):
            print("M=0, no cavity collapse operators added")
            return 0
        self.n_th = n_th
        c_ops = self.c_ops
        a = self.a
        rate = kappa*(n_th + 1)
        if(rate>0):
            c_ops.append(np.sqrt(rate)*a)
        rate = kappa*(n_th)
        if(rate>0):
            c_ops.append(np.sqrt(rate)*a.dag())
        self.kappa = kappa
        self.n_th = n_th
        self.c_ops = c_ops 
       # print("len of c_ops is : ",len(c_ops))
    
    def RtrnOps(self, indices, opts): 
        if(indices == 'all'):
            indices = range(self.N)
        r_ops = []
        sx_list = self.sx_list
        sy_list = self.sy_list
        sz_list = self.sz_list
        sm_list = self.sm_list
        sp_list = self.sp_list
        
        if(isinstance(indices, int) or isinstance(indices, float)):
            indices = [indices]

        for n in indices:
            if('sz' in opts): 
                r_ops.append(sz_list[n])
            if('spsm' in opts): 
                r_ops.append(sp_list[n]*sm_list[n])

        #cavity expectation
        if(self.M>0 and ('nphot' in opts)):
            a = self.a
            r_ops.append(a.dag()*a)
            
        self.r_ops = r_ops
        
    def ConstructH(self): # make hamiltonian
        H = 0;     N = self.N; w0 = self.w0     
        c_ops = self.c_ops
        sx_list = self.sx_list
        sy_list = self.sy_list
        sz_list = self.sz_list
        sm_list = self.sm_list
        sp_list = self.sp_list
        Jx = self.Jz 
        Jy = self.Jy
        Jz = self.Jz
        M = self.M
        
        if(M>0): 
            if(self.g == -1) :
                g = 0
                print('warning: g not specificed, setting to 0')
            else: g = self.g
            a = self.a;  g = self.g; omegac = self.omegac
            H += omegac*a.dag()*a 
            
        for n in range(N):
            H += - 0.5 * w0 * sz_list[n]
        for n in range(N-1):
            H += - 0.5 * Jx[n] * sx_list[n] * sx_list[n+1]
            H += - 0.5 * Jy[n] * sy_list[n] * sy_list[n+1]
            H += - 0.5 * Jz[n] * sz_list[n] * sz_list[n+1]
        if(M>0): 
            for n in range(N): 
                if isinstance(g, (int, float)):
                    H+= g * (a.dag() * sm_list[n] + sp_list[n]*a)
                elif isinstance(g, (list, np.ndarray)):
                    if len(g) != len(sm_list): print("g improper length")
                    H+= g[n] * (a.dag() * sm_list[n] + sp_list[n]*a)
                
        self.H = H
    def ConstructPsi0(self, first = 'excited'):
        psi_list = []
        
        if(first == 'excited'):
            psi_list.append(basis(2,1))
        elif(first == 'ground'):
            psi_list.append(basis(2,0))
            #print("excited changed to ground")
        for n in range(self.N-1):
            psi_list.append(basis(2,0))
        
        if(self.M>0):
            psi_list.append(basis(self.M))
        
        psi0 = tensor(psi_list)
        self.psi0 = psi0
        
    def expectH(self, tlist = np.linspace(0,50,200), rtrnopt = 'expect', solver = 'mesolve', ntraj = -1, verbose = False): #time evolve hamiltonian 
        #make psi0 : *any changes to initial state must be rewritten here*
        self.ConstructPsi0()
        psi0 = self.psi0
        H = self.H
        c_op_list = self.c_ops
        rop_list = self.r_ops
        
        solveopts = {}
        if(verbose):
            solveopts.update(progress_bar = True)
        else : 
            solveopts.update(progress_bar = None)
        if (solver == 'mcsolve') and (ntraj!=-1):
            solveopts.update(ntraj = ntraj)
        
        if(solver == 'mesolve'):
            print("mesolving...")
            result = mesolve(H, psi0, tlist, c_op_list, rop_list, **solveopts)
            return result.expect
        elif(solver == 'mcsolve'):
            print("mcsolving...")
            result = mcsolve(H, psi0, tlist, c_op_list, rop_list, **solveopts)
            return result
        elif(solver == 'steadystate'):
            if(verbose): print("finding steadystate...")
            rho = steadystate(H, c_op_list)
            result = [expect(op,rho) for op in rop_list]
            return result
 
    def GetInfo(self): #print out the current parameters
        lname = ["N", "M", "Jx", "Jy", "Jz", "gamma", "pump","num spin collapse ops", "num return ops", "H dims", "coupling, g", "kappa"]
        infolist = [self.N, self.M, self.Jx, self.Jy, self.Jz, self.gamma, self.pump,len(self.c_ops), len(self.r_ops), self.H.shape]
        if(self.M>0): 
            infolist+= [self.g,self.kappa]
        for ind,i in enumerate(infolist):
            width = max(len(n) for n in lname)
            print(lname[ind], end = ' '*(width - len(lname[ind])+3))
            if(isinstance(i,float)): print('%.3g'%i)
            elif(isinstance(i, int)): print('%d'%i)
            elif(isinstance(i, str)): print('%s'%i)
            elif(isinstance(i, np.ndarray)):
                if(i[0]==i[-1]): #same value repeating
                    if(isinstance(i[0],float)): print("[%.3f, ....]"%i[0])
                    elif(isinstance(i[0], int)): print("[%d, ....]"%i[0])
                else:
                    print(i)
            elif(isinstance(i, list)):
                if(i[0]==i[-1]): #same value repeating
                    if(isinstance(i[0],float)): print("[%.3f, ....]"%i[0])
                    elif(isinstance(i[0], int)): print("[%d, ....]"%i[0])
                else:
                    print(i)
            else:
                print(i)


# In[ ]:




