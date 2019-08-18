#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
class PhotoDetector:
    InitOpts = {'RtrnOps':
                    {'indices':'all', 'opts':['spsm', 'nphot']}}
    defaults =         {'N':6, #number of spin sites (length of chain)
         'M':10, #number of cavity occupation states
         'J':.1*2*np.pi, #spin-spin coupling 
         'g':.5, #spin-cavity coupling
         'pump1':.1, #first site pumping rate 
         'gammaN':.1, #terminal site spin dissipation
         'kappa':.3} #cavity dissipation 
     
    def __init__(self, pdict = defaults, makeH = True, GetInfo = False, kind = "II", initopts = None):
        if(initopts == None): initopts = PhotoDetector.InitOpts
    # define parameters with defaults
        defp = PhotoDetector.defaults.copy()
    # update with given dict, pdict
        self.defdict = defp.copy()      # first, set to defaults (avoiding changing defaults)
        for p in pdict: 
            if(p in defp): 
                self.defdict.update({p:pdict.get(p)})
        self.N, self.M, self.J, self.g,        self.pump1, self.gammaN, self.kappa = (self.defdict.get(i) for i in self.defdict)
    
    # update again for specificed system type
        self.SysUpdate(kind = kind)
        
    # prepare operator lists
        si = qeye(2)
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()
        sm = destroy(2)
        sp = create(2)
        bi = qeye(self.M)
        
        self.si_list, self.sx_list, self.sy_list, self.sz_list, self.sp_list, self.sm_list = ([] for _ in range(6))
        
        for n in range(self.N):
            op_list = []
            for m in range(self.N):
                op_list.append(si)
          
            if(self.M>0): op_list.append(bi)   

            op_list[n] = sx
            self.sx_list.append(tensor(op_list))

            op_list[n] = sy
            self.sy_list.append(tensor(op_list))

            op_list[n] = sz
            self.sz_list.append(tensor(op_list))

            op_list[n] = sp
            self.sp_list.append(tensor(op_list))

            op_list[n] = sm
            self.sm_list.append(tensor(op_list))
            
            op_list[n] = si
        
    # cavity mode operator
        if(self.M>0):
            aop = destroy(self.M)
            op_list[self.N] = aop
            self.a = tensor(op_list)
            
    # other default parameters
        self.n_th = 0.0 #for thermal bath
        self.w0 = 1.0*2*np.pi #atom frequency
        self.omegac = self.w0 #cavity frequency
        self.H = 0
        self.tdepargs = 0
        
    # set default J_lists
        self.setJconst(self.J, self.J, 0)
        
    # set default c_ops
        self.c_ops = []
        self.gamma = self.VariantCOp(-1, self.gammaN, self.sm_list)
        self.pump = self.VariantCOp(0, self.pump1, self.sp_list)
        if(self.M>0): self.kappa = self.VariantCOp(0, self.kappa, self.a)
        
    # set default r_ops : N sz ops and (if cavity) a^{\dag} a  
        self.r_ops = []
        self.RtrnOps(**initopts['RtrnOps'])
        
    # set initial state
        self.SetPsi0()
        
    # construct Hamiltonian
        if(makeH): self.ConstructH()
    
        if(GetInfo): self.GetInfo()
        
        
     #-----------------------------------  end of __init__   ------------------------------------------------   
    #adjust dictionary for specific system
    def SetParamfromDict(self, usedict):
        self.N, self.M, self.J, self.g,         self.pump1, self.gammaN, self.kappa = (usedict.get(i) for i in usedict)
        
    def SysUpdate(self,kind):
        if(kind not in ["I", "II", "III"]): print("SysUpdate: given 'kind' not compatible, (options are 'I', 'II', 'III')")
        ndict = self.defdict.copy()
        if(kind == "I"):
            ndict.update(M=0) 
        if(kind == "II"):
            if not isinstance(ndict['g'], (int, float)): print("warning: g has not been reset to single val"); ndict.update(g = 0.001)
        if(kind == "III"):
            if not isinstance(ndict['g'], (int,float)): print("warning: g has not been reset to single val"); ndict.update(g = 0.001)
            gtemp = [0,]*(self.N-1)+[self.g]
            ndict.update(g = gtemp)
        
        self.defdict = ndict.copy()
        self.SetParamfromDict(ndict)
    
    def setJconst(self, jx, jy, jz):
        self.Jx = jx *np.ones(self.N)
        self.Jy = jy  *np.ones(self.N)
        self.Jz = jz  *np.ones(self.N)
        
    #FOR TIME-INDEPENDENT COLLAPSE OPERATORS
    def VariantCOp(self, indices, vals, oplist): 
        # indices : location in array where collapse coefficient is nonzero
        # vals : values correspoding to indices
        arr = np.zeros(self.N)
        
        #given a single operator (ex: cavity operator)
        if not isinstance(oplist, (list, np.ndarray)):
            if isinstance(indices, (int, float)) and isinstance(vals, (int,float)):
                self.c_ops.append(np.sqrt(vals)*oplist)
                return vals
            
        
        #given list of indices
        if(isinstance(indices, (list, np.ndarray))):
            if len(indices)>len(oplist):
                print("VariantCOp: indices list larger than oplist"); return None
            
            if isinstance(vals, (int, float)): 
                # same val for all indices
                for i in range(len(indices)):
                    arr[indices[i]] = vals
                    self.c_ops.append(np.sqrt(vals)*oplist[i])
                    
            elif isinstance(vals, (list, np.ndarray)): 
                #list of values corresponding to indices
                if (len(indices)!=len(vals)) or (len(indices)>self.N):
                    print("VariantCOp: specificed nonzero indices and/or nonzero values have incorrect shape")
                    return None
                else:
                    for i in range(len(indices)):
                        arr[indices[i]] = vals[i]
                        self.c_ops.append(np.sqrt(vals[i])*oplist[i])
            else: 
                print("Don't understand vals type")
                return None
            
        #given a single indice
        elif isinstance(indices, (int, float)): 
            
            if isinstance(vals, (int,float)): 
                # must be given single value
                arr[indices] = vals
                self.c_ops.append(np.sqrt(vals)*oplist[indices])
                
            else: 
                print("VariantCOp: indices is an int or float, vals must also be an int or float")
                return None
        
        return arr #returns the c_op coefficients as an array 

    def TDepPump(self,func, args = 0):
        self.tdepargs = args
        c_ops = []
        c_ops.append(self.c_ops[0]) #gamma
        if(self.M>0): c_ops.append(self.c_ops[2]) #kappa
        c_ops.append([self.sp_list[0], func])
        self.c_ops = c_ops
            
    def RtrnOps(self, indices = 'all', opts = ['sz', 'nphot']):
        # indices: 'all' or [list of values] for spin sites,      opts: 'opt' or [list of 'opt'] 
        # possible opts:     'nphot' - include a^{\dagger} a     'sz'/'spsm'- type of spin expectation    
        
        #make non-list type indices iterable
        if(indices == 'all'):
            indices = range(self.N) 
        if(isinstance(indices, int) or isinstance(indices, float)):
            indices = [indices]

        for n in indices:
            if('sz' in opts): 
                self.r_ops.append(self.sz_list[n])
            if('spsm' in opts): 
                self.r_ops.append(self.sp_list[n]*self.sm_list[n])

        #cavity expectation
        if(self.M>0 and ('nphot' in opts)):
            self.r_ops.append(self.a.dag()*self.a)
    
    def SetPsi0(self,first = 'excited'):
        psi_list = []
        
        if(first == 'excited'):
            psi_list.append(basis(2,1))
        elif(first == 'ground'):
            psi_list.append(basis(2,0))
        for n in range(self.N-1):
            psi_list.append(basis(2,0))
        
        if(self.M>0):
            psi_list.append(basis(self.M))
        
        psi0 = tensor(psi_list)
        self.psi0 = psi0
        
    def ConstructH(self): 
        
        H = 0
            
        for n in range(self.N):
            H += - 0.5 * self.w0 * self.sz_list[n] 
            
        for n in range(self.N-1):
            H += - 0.5 * self.Jx[n] * self.sx_list[n] * self.sx_list[n+1]
            H += - 0.5 * self.Jy[n] * self.sy_list[n] * self.sy_list[n+1]
            H += - 0.5 * self.Jz[n] * self.sz_list[n] * self.sz_list[n+1]
            
        if(self.M>0): 
            H += self.omegac*self.a.dag()*self.a
            
            for n in range(self.N): 
                if isinstance(self.g, (int, float)): # if g is a single value, assumes inform coupling
                    H+= self.g * (self.a.dag() * self.sm_list[n] + self.sp_list[n]*self.a)
                elif isinstance(self.g, (list, np.ndarray)): # if g is list, must be same size as N
                    if len(self.g) != len(self.sm_list): print("ConstructH: g improper length")
                    H+= self.g[n] * (self.a.dag() * self.sm_list[n] + self.sp_list[n]*self.a)
                
        self.H = H
    
    def expectH(self, tlist = np.linspace(0,50,200), rtrnopt = 'expect',                 solver = 'mcsolve', ntraj = 500, odeopts = {}, verbose = False, reuse = False): #time evolve hamiltonian
        
        #set solver options
        solveopts = {}
        if(verbose):
            solveopts.update(progress_bar = True)
        else: 
            solveopts.update(progress_bar = None)
        if (solver == 'mcsolve'):
            solveopts.update(ntraj = ntraj)
        if(self.tdepargs !=0):     # if a time-dependent function with arguments is defined, include args in solve()
            solveopts.update(args = self.tdepargs)
            if(verbose): print(solveopts['args'])
            
        if(len(odeopts)>0):
            if(solver !='mcsolve'): print("odeopts only set up for mcsolve, must go in and add to class")
            else:
                opts = Options(**odeopts)
                print("solving with options. Dictionary:", odeopts)
                result = mcsolve(self.H, self.psi0, tlist, self.c_ops, self.r_ops,options = opts,**solveopts)
                return result
                
        if(solver == 'mesolve'):
            print("mesolving...")
            result = mesolve(self.H, self.psi0, tlist, self.c_ops, self.r_ops,**solveopts)
            if not verbose: print("done")
        elif(solver == 'mcsolve'):
            print("mcsolving...")
            result = mcsolve(self.H, self.psi0, tlist, self.c_ops, self.r_ops,**solveopts)
            if not verbose: print("done")
        elif(solver == 'steadystate'):
            print("finding steadystate...", end = '\t')
            rho = steadystate(self.H, self.c_ops)
            result = [expect(op,rho) for op in self.r_ops]
            print("done")
            
        return result
    
                
    def GetInfo(self):
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




