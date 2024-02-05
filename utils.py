import torch


def generate_betas(T, beta_0=0.0001, beta_T=1):
    # betas = torch.zeros(T)
    # s = 0.008
    # for t in range(T):
    #     betas[t] = torch.cos((t/T + s)/(1 + s) * torch.pi / 2)

    betas = torch.linspace(beta_0, beta_T, T) # linear for now
    return betas



def compute_transition_matrix(betas, T, num_bins=4):
    transition_matrices = torch.zeros((T, num_bins, num_bins))

    for t in range(T):
        beta = betas[t]
        transition_matrix = torch.zeros((num_bins, num_bins))

        for i in range(num_bins):
            for j in range(num_bins):
                if i == j:
                    transition_matrix[i, j] = 1 - (num_bins - 1)/num_bins * beta
                else:
                    transition_matrix[i, j] = 1/num_bins * beta
        
        transition_matrices[t] = transition_matrix

    return transition_matrices