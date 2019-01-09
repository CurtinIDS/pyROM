import scipy.io as sio
import scipy.sparse as sparse
import numpy as np
import copy
from matplotlib import animation
from itertools import product
from bokeh.plotting import figure
import time


class Domain():
    '''
    Domain for POD and DMD simulation
    '''
    def __init__(self,mat_file,timesteps=100,total_time=1,cutoff=10,snapshots=25,perturbation=None):
        self.wall=time.time() #Start timer to compute walltime
        tmp=copy.deepcopy(locals()) #Don't copy list to another list!
        for each in tmp:
            if not "__" in each:
                exec ("self."+each+"="+each)

        Matrix=sio.loadmat(mat_file) #Loads matrix in Matlab format
        
        for each in Matrix:
            if not "__" in each:
                exec ("self."+each+"=Matrix['"+each+"']")
                if "scipy.sparse" in str(type(Matrix[each])):
                    exec("self."+each+"=sparse.bsr_matrix(self."+each+")") #Convert all matrices to bsr form
                else:
                    exec("self."+each+"=np.matrix(self."+each+")")
        
        
        #Check perturbation
        
        self.dt=total_time/timesteps
        
        self.m=self.A.shape[0]
        self.length=np.sqrt(self.m)
        self.h=np.ones([1,3])*1/(self.length-1)  #Generalize in future
        

        self.hx=1/(self.length-1)
        self.hy=1/(self.length-1)
        
        
        self.xf=np.linspace(0,1,1+(1/self.hx))
        self.yf=np.linspace(0,1,1+(1/self.hy))

        
        self.u0=np.zeros([self.m,1]) #Initial u
        
        #Compute source/drain perturbation
        inp=[]
        for j in range(self.yf.shape[0]):
            for i in range(self.xf.shape[0]):
                inp.append([self.xf[i],self.yf[j]])
        
        out=map(perturbation,inp)
        
        
        self.u0=np.matrix(list(out)).transpose() #Don't list if using map/reduce in parallel

        correct = lambda x:x-1 #Indices in MATLAB start from 1
        
  
        self.uc=np.linalg.solve(self.R_enrich*self.R_enrich.transpose(),(self.R_enrich*self.u0))
      
        #Set BCs
        nright=np.vectorize(correct)(self.nright)
        nleft=np.vectorize(correct)(self.nleft)
        ntop=np.vectorize(correct)(self.ntop)
        nbottom=np.vectorize(correct)(self.nbottom)
    
        #Fine scale
        for i in np.array([nright,nleft,ntop,nbottom]).flatten():
            for each in i:
                self.uc[each]=0
       
        #self.u0_=self..sz.transpose()*self.uc
        self.m=copy.deepcopy(self.Ams.shape[0])
        self.A=copy.deepcopy(self.Ams)
        self.M=copy.deepcopy(self.Mms)
        self.u0=copy.deepcopy(self.uc)
        self.F=copy.deepcopy(self.Fms)
        self.shape=[int(self.length),int(self.length)] #change this later for 3d
        self.wall=str(np.round(time.time()-self.wall,2))+" s"  #Total walltime
        
    def march(self):
        S=np.linalg.solve(self.M+(self.dt*self.A),self.M)
        G=np.linalg.solve(self.M+(self.dt*self.A),self.F*self.dt)
        V=np.matrix(np.zeros([self.timesteps,self.u0.shape[0]]))
        V[0,:]=self.u0.transpose()
        for i in range(1,self.timesteps):
            V[i,:]=(S*(V[i-1,:].transpose())+G).transpose()
        V=V.transpose()
        
        
        self.nstart=0
        Q1=np.copy(V[:,self.nstart:self.nstart+self.snapshots])
        Q2=np.copy(V[:,self.nstart+1:self.nstart+self.snapshots+1])
        
        
        [v,d]=np.linalg.eig(S)
        ind = np.argsort(v)
        d=d[:self.m,ind]  #Check this <<<-------
        self.Q1=Q1
        self.Q2=Q2
        self.S=S
        self.d=d
        self.V=V
        self.G=G
       
    def dmd(self,cutoff=10):
        #Compute walltime
        time_start=time.time()
        
        Uu,Ss,Ww=np.linalg.svd(self.Q1,full_matrices=False)
        
        Ss=np.diag(Ss)
        Ssinv=np.matrix(np.linalg.inv(Ss))
        Ss=np.matrix(Ss)
        Ww=np.matrix(Ww)
        Uv=np.matrix(Uu)
        self.Q2=np.matrix(self.Q2)
        

        S_tilda=Uu.transpose()*self.Q2*Ww*Ssinv
        mu,Y=np.linalg.eig(S_tilda)
        ind=np.argsort(mu)
        mu=mu[ind]
        mu=np.matrix(mu[:self.m])
        
        DM1=Uu*Y
        DM2=DM1[:self.m,ind]
        DM=DM2[:,:cutoff]
        solxDM=np.matrix(np.zeros([DM.shape[1],self.timesteps])) #Z's shape is DM.shape[1]
        solDM=np.matrix(np.zeros([self.V.shape[0],self.timesteps]))
        for i in range(self.timesteps):
            Z=DM.transpose()*DM
            xDM=np.linalg.solve(Z,DM.transpose()*self.V[:,i])
            solxDM[:,i]=xDM
            uDM=DM*xDM
            solDM[:,i]=uDM
        MD=DM.transpose()*DM
        SD=DM.transpose()*self.S*DM
        SDMD=np.linalg.solve(MD,SD)
        FDMD=np.linalg.solve(MD,DM.transpose()*self.G)
        uDMD=np.matrix(np.zeros([self.timesteps,Z.shape[0]]))
        uDMDFine=np.matrix(np.zeros([self.timesteps,DM.shape[0]]))
        uDMD[self.nstart,:]=np.linalg.solve(Z,(DM.transpose()*self.V[:,self.nstart])).transpose()
        #Evolve
        for i in range(self.nstart+1,self.timesteps):
            uDMD[i,:]=(SDMD*(uDMD[i-1,:].transpose())+FDMD).transpose()
            uDMDFine[i,:]=(DM*uDMD[i,:].transpose()).transpose()
        
        uDMDms_fine = uDMDFine * self.R_enrich
        Vmsfine = self.R_enrich.transpose() * self.V
        self.DMD=Vmsfine
        self.DMD.wall=str(np.round(time.time()-time_start,2))+" s"
        
    def pod(self,cutoff=10):
        time_start=time.time()
        UP,SP,WP=np.linalg.svd(self.Q1,full_matrices=False)
        PM1=np.matrix(UP[:,:cutoff])
        MP=np.matrix(PM1.transpose())*PM1
        SP=PM1.transpose()*self.S*PM1
        SPOD=np.linalg.solve(MP,SP)
        muPOD,XPOD=np.linalg.eig(SPOD)
        muPOD2=-np.sort(-muPOD) #Sort it by asc. eigen values 
        index2=np.argsort(-muPOD) #Indices corresponding to the sorted 
        PM=PM1[:self.m,index2]
        DA,XA=np.linalg.eig(self.S)
        dda=-np.sort(-DA)
        ii=np.argsort(-DA)
        EVA=XA[:,ii]
        solxPM=np.matrix(np.zeros([PM.shape[1],self.timesteps])) #ZP's shape is DM.shape[1]
        solPM=np.matrix(np.zeros([self.V.shape[0],self.timesteps]))
        
        for i in range(self.timesteps):
            ZP=PM.transpose()*PM;
            xPM=np.linalg.solve(ZP,(PM.transpose()*self.V[:,i]))
            solxPM[:xPM.shape[0],i]=xPM
            uPM=PM*xPM
            solPM[:self.V.shape[0],i]=uPM
        MP=PM.transpose()*PM;
        SP=PM.transpose()*self.S*PM;
        SPOD=np.linalg.solve(MP,SP)
        uPOD=np.matrix(np.zeros([self.timesteps,ZP.shape[0]]))
        uPODFine=np.matrix(np.zeros([self.timesteps,PM.shape[0]]))
        FPOD=np.linalg.solve(MP,PM.transpose())*self.G
        uPOD[self.nstart,:]=np.linalg.solve(ZP,(PM.transpose()*self.V[:,self.nstart])).transpose()
        for i in range(self.nstart+1,self.timesteps):
            uPOD[i,:]=(SPOD*uPOD[i-1,:].transpose()+FPOD).transpose()
            uPODFine[i,:]=(PM* uPOD[i,:].transpose()).transpose()
        uPODms_fine = uPODFine * self.R_enrich;
        self.POD= self.R_enrich.transpose() * self.V
        self.POD.values=self.R_enrich.shape
        self.POD.wall=str(np.round(time.time()-time_start,2))+" s"

    def bokeh_out(*args,axis=0,timesteps=200):
        #The last axis is always timestep
        cnt=0
        length=0
        stack_of_images=[]
        for each in args:
            if type(each)==np.matrix:
                cnt+=1
                min_timestep=min(10,each.shape[len(each.shape)-1])
                stack_of_images.append(np.copy(each))
            elif type(each)==Domain:
                length=np.copy(each.shape)
        min_length=np.prod(length)
        orig_length=np.copy(length)
        length[axis]=cnt*length[axis]
        tmp_matrix=np.zeros([orig_length[0]*cnt,min_timestep*orig_length[1]])
        maj_count=0
        for i in np.linspace(0,timesteps,min_timestep):
            cnt=0
            for j in range(len(stack_of_images)):
                tmp_img=stack_of_images[j][:,int(i)].reshape(min_length)
                tmp_img=tmp_img/(np.max(tmp_img)-np.min(tmp_img))
                tmp_matrix[j*orig_length[0]:(j+1)*orig_length[0],maj_count*orig_length[1]:(maj_count+1)*orig_length[1]]=tmp_img.reshape(orig_length)
                cnt+=1
            maj_count+=1
        delmeimg=tmp_matrix.reshape([cnt*orig_length[0],min_timestep*orig_length[1]],order='F')
        p = figure(x_range=(0, 1), y_range=(0, 5))
        p.image(image=[delmeimg],x=0,y=0,dw=6,dh=3,palette="Spectral11")

        return p

