import torch
import torch.nn.functional as F



def sample_categorical(logits, uniform_noise):
    """Samples from a categorical distribution.

    Args:
        logits: logits that determine categorical distributions. Shape should be
        broadcastable under addition with noise shape, and of the form (...,
        num_classes).
        uniform_noise: uniform noise in range [0, 1). Shape: (..., num_classes).

    Returns:
        samples: samples.shape == noise.shape, with samples.shape[-1] equal to
        num_classes.
    """
    uniform_noise = torch.clip(
        uniform_noise, min=torch.finfo(uniform_noise.dtype).tiny, max=1.)
    gumbel_noise = -torch.log(-torch.log(uniform_noise))
    sample = torch.argmax(logits + gumbel_noise, dim=-1)
    return F.one_hot(sample, num_classes=logits.size(-1))


def categorical_kl_logits(logits1, logits2, eps=1.e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
        logits1: logits of the first distribution. Last dim is class dim.
        logits2: logits of the second distribution. Last dim is class dim.
        eps: float small number to avoid numerical issues.

    Returns:
        KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = (
        F.softmax(logits1 + eps, dim=-1) *
        (F.log_softmax(logits1 + eps, dim=-1) -
         F.log_softmax(logits2 + eps, dim=-1)))
    return torch.sum(out, dim=-1)


def categorical_kl_probs(probs1, probs2, eps=1.e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
        probs1: probs of the first distribution. Last dim is class dim.
        probs2: probs of the second distribution. Last dim is class dim.
        eps: float small number to avoid numerical issues.

    Returns:
        KL(C(probs) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps))
    return torch.sum(out, dim=-1)


def categorical_log_likelihood(x, logits):
    """Log likelihood of a discretized Gaussian specialized for image data.

    Assumes data `x` consists of integers [0, num_classes-1].

    Args:
        x: where to evaluate the distribution. shape = (bs, ...), dtype=int32/int64
        logits: logits, shape = (bs, ..., num_classes)

    Returns:
        log likelihoods
    """
    log_probs = F.log_softmax(logits)
    x_onehot = F.one_hot(x, num_classes=logits.size(-1))
    return torch.sum(log_probs * x_onehot, dim=-1)


def meanflat(x):
    """Take the mean over all axes except the first batch dimension."""
    return x.mean(dim=tuple(range(1, x.dim())))


