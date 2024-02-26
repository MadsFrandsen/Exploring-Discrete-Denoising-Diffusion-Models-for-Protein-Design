import torch
import torch.nn.functional as F
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
                transition_bands, model_prediction, model_output, torch_dtype=torch.float32):
        
        self.model_prediction = model_prediction # x_start, xprev
        self.model_output = model_output # logits or logistic_pars
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
    

    def _get_logits_from_logistic_pars(self, loc, log_scale):

        loc = torch.unsqueeze(loc, dim=-1)
        log_scale = torch.unsqueeze(log_scale, dim=-1)

        inv_scale = torch.exp(-(log_scale - 2.))

        bin_width = 2. / (self.num_pixel_vals - 1.)
        bin_centers = torch.linspace(start=-1., stop=1., steps=self.num_pixel_vals,
                                     device=loc.device)
        
        for _ in range(loc.ndim - 1):
            bin_centers = bin_centers.unsqueeze(0)
        
        bin_centers = bin_centers - loc
        log_cdf_min = F.logsigmoid(
            -inv_scale * (bin_centers - 0.5 * bin_width))
        log_cdf_plus = F.logsigmoid(
            -inv_scale * (bin_centers + 0.5 * bin_width))
        
        logits = None
        return logits


    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """Compute logits of q(x_{t-1} | x_t, x_start) in PyTorch."""

        if x_start_logits:
            assert x_start.shape == x_t.shape[:-1] + (self.num_pixel_vals,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        if x_start_logits:
            # PyTorch's softmax operates on the specified dimension.
            fact2 = self._at_onehot(self.q_mats, t-1, F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t-1, x_start)
            # For one_hot encoding and addition of eps for numerical stability
            tzero_logits = torch.log(F.one_hot(x_start.to(torch.int64), num_classes=self.num_pixel_vals).to(torch.float) + self.eps)

        # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
        # where x_{-1} == x_start. This should be equal to the log of x_0.
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        t_broadcast = t.unsqueeze(1).expand(-1, *out.shape[1:])
        return torch.where(t_broadcast == 0, tzero_logits, out)
    

    def p_logits(self, model_fn, *, x, t):
        assert t.shape == (x.shape[0],)
        model_output = model_fn(x, t)

        if self.model_output == 'logitsl':
            model_logits = model_output
        
        elif self.model_output == 'logistic_pars':
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        
        else:
            raise NotImplementedError(self.model_output)
        
        if self.model_prediction == 'x_start':
            pred_x_start_logits = model_logits

            t_broadcast = t.unsqueeze(1).expand(-1, *model_logits.shape[1:])
            model_logits = torch.where(t_broadcast == 0,
                                       pred_x_start_logits,
                                       self.q_posterior_logits(pred_x_start_logits, x,
                                                               t, x_start_logits=True)
                                        )
        
        elif self.model_prediction == 'xprev':
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)
        
        assert (model_logits.shape == 
                pred_x_start_logits.shape == x.shape + (self.num_pixel_vals,))
        return model_logits, pred_x_start_logits
    

    def p_sample(self, model_fn, *, x, t, noise):
        model_logits, pred_x_start_logits = self.p_logits(
            model_fn=model_fn, x=x, t=t)
        assert noise.shape == model_logits.shape, noise.shape

        # No noise when t == 0
        nonzero_mask = (t != 0).astype(x.dtype).reshape(x.shape[0],
                                                        *([1] * (len(x.shape))))
        noise = torch.clip(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))

        sample = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)

        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape
        return sample, F.softmax(pred_x_start_logits, dim=-1)


    def p_sample_loop(self, model_fn, *, shape, rng_seed,
                      num_timesteps=None, return_x_init=False):
        torch.manual_seed(rng_seed)
        rng = torch.Generator()
        rng.manual_seed(rng_seed)

        noise_shape = shape + (self.num_pixel_vals,)

        def body_fun(i, x):
            t = torch.full([shape[0]], self.num_timesteps - 1 - i)
            x, _ = self.p_sample(
                model_fn=model_fn,
                x=x,
                t=t,
                noise=torch.rand(size=noise_shape, generator=rng)
            )
            return x
        
        if self.transition_mat_type in ['gaussian', 'uniform']:
            x_init = torch.randint(low=0, high=self.num_pixel_vals,
                                   size=shape, generator=rng)
        elif self.transition_mat_type == 'absorbing':
            x_init = torch.full(size=shape, fill_value=self.num_pixel_vals//2,
                                dtype=torch.int32)
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                f", but is {self.transition_mat_type}"
            )
        
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        
        final_x = x_init
        for i in range(num_timesteps):
            final_x = body_fun(i, final_x)
        
        assert final_x.shape == shape
        if return_x_init:
            return x_init, final_x
        else:
            return final_x





