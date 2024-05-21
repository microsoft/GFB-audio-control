import torch
import numpy as np
from tqdm import tqdm
import einops
import ot as pot
from functools import partial

def calculate_curvature(trajectory):
    #as used in the paper, just for reference
    base=trajectory[0]-trajectory[-1]
    base=base.reshape(base.shape[0], -1)
    N=len(trajectory)
    dt=1.0/N
    mse=[]
    for i in range(1,N):
        v=(trajectory[i-1]-trajectory[i])/dt
        v=v.reshape(v.shape[0], -1)
        mse.append(torch.mean((v-base)**2, dim=-1).cpu())
    return torch.mean(torch.stack(mse)), mse

class OTCFM():
    """
    Class that takes care of all diffusion-related stuff. This includes training losses, sampling, etc.
    """
    def __init__(self,
        cfg_value=2,
        num_cond_params=2,
        minibatch_OT=False,
        minibatch_OT_args=None,
        order=1,
        ) -> None:
        self.cfg_value=cfg_value
        self.num_cond_params=num_cond_params
        self.minibatch_OT=minibatch_OT
        self.minibatch_OT_args=minibatch_OT_args
        self.order=order
        if self.minibatch_OT:
            if self.minibatch_OT_args.algorithm=="emd":
                self.ot_fn=partial(pot.emd, numItermax=1e5, numThreads=1)
            elif self.minibatch_OT_args.algorithm=="sinkhorn":
                self.ot_fn=partial(pot.sinkhorn, reg=0.001, numItermax=int(1e2),method='sinkhorn_log')

    def get_train_tuple(self,x,z,t):

        #linear interpolation
        x_t= t*z+(1.-t)*x

        #CFM objective
        target= z-x
        return x_t, target


    def OT_plan(self,x,z):
        #divide the pairwise L2 operation to save memory
        #compute pairwise L2 matrix
        M=((x[None,:,:] - z[:,None,:])**2).mean(-1)

        M=M.T
        M+=1e-5 #for numerical stability

        #assert that M is a square matrix
        assert M.shape[0]==M.shape[1]
        a, b = pot.unif(M.shape[0]), pot.unif(M.shape[1])
        a=torch.from_numpy(a).float().to(M.device)
        b=torch.from_numpy(b).float().to(M.device)
            
        #apply the solver
        P=self.ot_fn(a,b,M)

        if self.minibatch_OT_args.algorithm=="emd":
            #in the case of deterministic OT, this is equivalent to what we do below, and probably faster
            index=P.max(axis=1).indices
        else:
            P*=x.shape[0]
            P/=P.sum(-1)
            normalized_P = P / P.sum(dim=1, keepdim=True)
            index = torch.multinomial(normalized_P, 1, replacement=True).squeeze()

        return index

    def get_audio_noise_pairs(self,x, x_plan):
        B,C,T=x.shape
        assert C==1, "C must be 1"
        #divide the audio in chunks
        x_chunk=einops.rearrange(x_plan.squeeze(1), "b (t c) -> (b t) c", c=self.minibatch_OT_args.chunk_size)

        z=torch.randn((x_chunk.shape[0], self.minibatch_OT_args.chunk_size), device=x.device)

        index=self.OT_plan(x_chunk, z)
        z=z[index]
    
        z=einops.rearrange(z, "(b t) c -> b (t c)", b=B, c=self.minibatch_OT_args.chunk_size)

        z=z.unsqueeze(1)     
        return x, z

    def compute_loss(self, x, model=None, cond=None,  task="reverb"):
        assert model is not None, "model must be provided"

        loss_dict={}

        if task=="reverb":
            T60=cond["T60"]
            C50=cond["C50"]
            used_conds=[T60, C50]
        elif task=="declipping":
            SDR=cond["SDR"]
            used_conds=[SDR]
        else:
            raise NotImplementedError("the task {} is not implemented".format(task))
        
        if self.minibatch_OT:
            x_plan=x
            x,z=self.get_audio_noise_pairs(x, x_plan)
        else:
            #classic training with independent noise
            z=torch.randn_like(x)
    
        B= x.shape[0]
        t=torch.rand((B), device=x.device)
        eps=1e-6
        t = t * (1 - eps) + eps
        #t must lie in (0,1)
        assert torch.all(t>0) and torch.all(t<1)

        x_t, target = self.get_train_tuple(x,z,t.unsqueeze(1).unsqueeze(1))

        if cond is not None:
            cond_tensor=[c.to(x.device).to(x.dtype).unsqueeze(-1) for c in used_conds]
            #concatenate both in a single tensor
            cond_tensor=torch.cat(cond_tensor, dim=1)
            #with a certain probability, set all the condition to some predefined value 
            dropped=torch.rand(cond_tensor.shape, dtype=cond_tensor.dtype,device=cond_tensor.device)<0.8
            cond_tensor=cond_tensor*dropped.float()+self.cfg_value*(1-dropped.float())


        if len(cond_tensor.shape)==3:
            cond_tensor=cond_tensor.squeeze(-1)
        if len(t)==1:
            t=t.unsqueeze(-1)
        pred=model(x_t, torch.log(t), cond=cond_tensor)

        loss = (target.clone() - pred)**2
        loss_dict["error"]=loss.detach()

        total_loss=loss.mean()

        return total_loss, loss_dict, t

    def get_schedule(self, Tsteps, end_t=1, type="linear"):
        if type=="linear":
            return torch.linspace(0, end_t, Tsteps+1)[1:]
        elif type=="cosine":
            pi=torch.tensor(np.pi)
            #if end_t!=1:
            #    raise NotImplementedError("end_t must be 1 for cosine schedule")
            t=torch.linspace(0,1,Tsteps+1)
            base=0.5*((1+ torch.cos(t*pi+ pi)))

            #cut it at end_t (find the index of the first element that is greater than end_t)
            base=base[base<=end_t]

            return base[1:]

    def sample_conditional(self, 
        shape, #B, C, T
        model, #DNN
        Tsteps, #number of steps
        cond=None, #conditioning parameters
        cfg=None, #classifier-free guidance
        same_noise=False, #if True, use the same noise for all the elements of the batch
        device="cuda",
        ):
        B, C, T = shape
        t=self.get_schedule(Tsteps).to(device)
        #sample prior
        if not same_noise:
            z=torch.randn(B,C,T).to(device)
        else:
            z=torch.randn(1,C,T).to(device)
            z=z.expand(B,C,T)

        xt=z
        for i in tqdm(reversed(range(0,Tsteps))):
            if i==0:
                xt=self.sampling_step(xt, t[i].expand(B,), t[i].expand(B,)*0, model, 1, cond=cond, cfg=cfg)
            else:
                xt=self.sampling_step(xt, t[i].expand(B,), t[i-1].expand(B,), model, self.order, cond=cond, cfg=cfg)
        return xt

    def sample_unconditional(self, 
        shape, #B, C, T
        model, #DNN
        Tsteps, #number of steps
        device,
        output_trajectory=False,
        ):
        B, C, T = shape
        t=self.get_schedule(Tsteps).to(device)
        #sample prior
        z=torch.randn(B,C,T).to(device)*t[-1]

        trajectory=[]
        denoised_estimates=[]
        xt=z
        trajectory.append(xt)
        for i in tqdm(reversed(range(0,Tsteps))):
            if i==0:
                xt, x0=self.sampling_step(xt, t[i].expand(B,), t[i].expand(B,)*0, model, 1, get_denoised_estimate=True)
                trajectory.append(xt)
                denoised_estimates.append(x0)
            else:
                xt, x0=self.sampling_step(xt, t[i].expand(B,), t[i-1].expand(B,), model, self.order, get_denoised_estimate=True)
                trajectory.append(xt)
                denoised_estimates.append(x0)
        if output_trajectory:
            return xt, trajectory,denoised_estimates
        else:
            return xt

    def model_call(self, model, x, t, cond=None):
        if cond is None:
            cond=torch.zeros((x.shape[0],self.num_cond_params), dtype=x.dtype, device=x.device)+self.cfg_value
        with torch.no_grad():
            v=model(x, torch.log(t), cond=cond)
        return v

    def model_call_cfg(self, model, x, t, cond=None, cfg=None):
        v_unc=self.model_call(model, x, t)
        v_cond=self.model_call(model, x, t, cond=cond)
        return (1-cfg)*v_unc+cfg*v_cond

    def sampling_step(self,
        xt, #noisy input
        t, #current timestep
        t2, #next timestep
        model, #DNN
        order=1, #1 for Euler, 2 for Heun
        cond=None, #conditioning parameters
        cfg=None, #classifier-free guidance
        get_denoised_estimate=False, #if True, return the denoised estimate
        ):
        if cond is not None:
            assert cfg is not None, "If cond is not None, cfg must be provided"
        if cond is None:
            vt = self.model_call(model,xt, t)
        else:
            vt = self.model_call_cfg(model,xt, t, cond=cond, cfg=cfg)

        #using 
        dt = (t2 - t).view(-1,1,1)
        #print(dt)

        if order==2:
            x2 = xt.detach().clone() + vt * dt
            if cond is None:
                vt_2 = self.model_call(model,x2, t2)
            else:
                vt_2 = self.model_call_cfg(model,x2, t2, cond=cond, cfg=cfg)
            vt = (vt + vt_2) / 2

        if get_denoised_estimate:
            denoised_estimate=xt.detach().clone() -vt *t.view(-1,1,1)

        xt = xt.detach().clone() + vt * dt

        if get_denoised_estimate:
            return xt, denoised_estimate
        else:
            return xt 


    def bridge(self, x0, model=None, Tsteps=30, cond=None, cfg=None, bridge_end_t=1, output_trajectory=False, schedule_type="linear"):
        """
        Diffusion bridge, where we apply unconditonal forward diffusion, and conditional backward diffusion
        args:
            x0: (1, C, T) tensor
            Tsteps: number of steps
            cond: (1, num_cond_params) tensor
        """
        assert model is not None, "model must be provided"
        device=x0.device

        t_schedule=self.get_schedule(Tsteps, end_t=bridge_end_t, type=schedule_type).to(device)
        if output_trajectory:
            trajectory={}
            trajectory["t"]=t_schedule
        #sample prior
        #maybe start with a small sigma
        x0= t_schedule[0]*torch.randn(x0.shape, device=device)+(1.-t_schedule[0])*x0
        if output_trajectory:
            z, traj=self.forward_ODE(x0, model=model, schedule=t_schedule, output_trajectory=output_trajectory)
            trajectory["forward"]=traj
        else:
            z=self.forward_ODE(x0, model=model, schedule=t_schedule, output_trajectory=output_trajectory)

        if cond is not None:
            B=cond.shape[0]
        else:
            B=x0.shape[0]
        #expand z to B
        zexp=z.expand(B, -1, -1)
        if output_trajectory:
            xnew, traj=self.backward_ODE(zexp, model=model, cond=cond, cfg=cfg, schedule=t_schedule, output_trajectory=output_trajectory)
            trajectory["backward"]=traj
        else:
            xnew=self.backward_ODE(zexp, model=model, cond=cond, cfg=cfg, schedule=t_schedule, output_trajectory=output_trajectory)

        if output_trajectory:
            return xnew, z, trajectory
        else:
            return xnew, z

    def forward_ODE(self, 
        xt, #B, C, T
        model=None, #DNN
        cond=None, #conditioning parameters
        cfg=None, #classifier-free guidance
        schedule=None, #schedule of timesteps
        output_trajectory=False,
        ):
        B, C, T = xt.shape
        Tsteps=schedule.shape[0]

        trajectory=[]
        trajectory.append(xt)
        for i in tqdm(range(0,Tsteps)):
            if i==Tsteps-1:
                xt=self.sampling_step(xt, schedule[i].expand(B,), schedule[i].expand(B,)*0+schedule[-1], model, 1, cond=cond, cfg=cfg)
                trajectory.append(xt)
            else:
                xt=self.sampling_step(xt, schedule[i].expand(B,), schedule[i+1].expand(B,), model, 2, cond=cond, cfg=cfg)
                trajectory.append(xt)

        if output_trajectory:
            return xt, trajectory
        else:
            return xt

    def backward_ODE(self, 
        xt, #B, C, T
        model=None, #DNN
        cond=None, #conditioning parameters
        cfg=None, #classifier-free guidance
        schedule=None, #schedule of timesteps
        output_trajectory=False,
        ):
        B, C, T = xt.shape
        Tsteps=schedule.shape[0]
        trajectory=[]
        trajectory.append(xt)
        for i in tqdm(reversed(range(0,Tsteps))):
            if i==0:
                xt=self.sampling_step(xt, schedule[i].expand(B,), schedule[i].expand(B,)*0, model, 1, cond=cond, cfg=cfg)
                trajectory.append(xt)
            else:
                xt=self.sampling_step(xt, schedule[i].expand(B,), schedule[i-1].expand(B,), model, self.order, cond=cond, cfg=cfg)
                trajectory.append(xt)

        if output_trajectory:
            return xt, trajectory
        else:
            return xt
