import torch
import numpy as np




def generate_betas(start: float, stop: float, num_steps: int, type: str):
    """Function to generate betas."""
    if type == 'linear':
        return torch.linspace(start, stop, num_steps)
    elif type == 'cosine':
        steps = (
            torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
        )
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        betas = torch.minimum( 1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        return betas
    elif type == 'jsd':
        return 1. / torch.linspace(num_steps, 1., num_steps)
    else:
        raise NotImplementedError(type)
    

class DiscreteDiffusion:
    """Discrete state space diffusion process."""

    def __init__(self, betas, transition_mat_type, num_bits, 
                transition_bands, torch_dtype=torch.float32):
        
        self.torch_dtype = torch_dtype

        self.num_bits = num_bits
        # Data \in {0, ..., num_pixel_vals-1}
        self.num_pixel_vals = 2**self.num_bits
        self.transition_bands = transition_bands
        self.transition_mat_type = transition_mat_type
        self.eps = 1.e-6

        self.betas = betas = betas.astype(torch.float64)
        self.num_timesteps = betas.shape[0]

        if self.transition_mat_type == 'uniform':
            q_one_step_mats = [self._get_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        else:
            raise ValueError(
                f"transition_mat_type must be 'uniform'"
                f", but is {self.transition_mat_type}"
            )
        self.q_onestep_mats = torch.stack(q_one_step_mats, dim=0)
        assert self.q_onestep_mats.shape == (self.num_timesteps,
                                             self.num_pixel_vals,
                                             self.num_pixel_vals)
        
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            q_mat_t = torch.tensordot(q_mat_t, self.q_onestep_mats[t],
                                      dims=[[1], [0]])
            q_mats.append(q_mat_t)
        self.q_mats = torch.stack(q_mats, dim=0)
        assert self.q_mats.shape == (self.num_timesteps, self.num_pixel_vals,
                                     self.num_pixel_vals), self.q_mats.shape
        
        self.transpose_q_onestep_mats = self.q_onestep_mats.permute(0, 2, 1)

        del self.q_onestep_mats

    def _get_full_transition_mat(self, t):
        beta_t = self.betas[t]
        mat = torch.full(size=(self.num_pixel_vals, self.num_pixel_vals),
                        fill_value=beta_t / float(self.num_pixel_vals),
                        dtype=torch.float64)
        diag_val = 1. - beta_t * (self.num_pixel_vals - 1.) / self.num_pixel_vals
        mat.fill_diagonal_(diag_val)
        return mat


    
    def _get_transition_mat(self, t: int):
        if self.transition_bands is None:
            return self._get_full_transition_mat(t)
        
        beta_t = self.betas[t]

        mat = torch.zeros((self.num_pixel_vals, self.num_pixel_vals),
                          dtype=torch.float64)
        off_diag = torch.full(size=(self.num_pixel_vals-1,),
                              fill_value=beta_t / float(self.num_pixel_vals),
                              dtype=torch.float64)
        for k in range(1, self.transition_bands + 1):
            mat += torch.diag(off_diag, diagonal=k)
            mat += torch.diag(off_diag, diagonal=-k)
            off_diag = off_diag[:-1]
        
        diag = 1. - mat.sum(1)
        mat += torch.diag(diag, diagonal=0)
        return mat
    
    
    def _at(self, a, t, x):
        a = a.to(dtype=self.torch_dtype)
        t_broadcast = t.unsqueeze(1).expand(-1, *x.shape[1:])
        
        return a[t_broadcast, x]
    
    
    def _at_onehot(self, a, t, x):
        a = a.to(dtype=self.torch_dtype)
        t_indexed = a[t]

        return torch.matmul(x, t_indexed.unsqueeze(1))
    

    def q_probs(self, x_start, t):
        return self._at(self.q_mats, t, x_start)
    

    def q_sample(self, x_start, t, noise):
        assert noise.shape == x_start.shape + (self.num_pixel_vals,)
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        noise = torch.clip(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, axis=-1)
    




