import torch
import numpy as np
from tqdm import tqdm
from constants import PROTEIN_ALPHABET, MSA_AAS, GAP, ALL_AAS


def cumprod_matrix(a):
    """
    Takes a list of transition matrices and outputs a list
    of the cumulative products (Q_bar) at each timestep
    """
    a_bar = [a[0]]
    start = a[0]
    for i in range(len(a) - 1):
        a_prod_temp = torch.mm(start, a[i + 1])
        start = a_prod_temp
        a_bar.append(a_prod_temp)
    return a_bar


def softmax(x):
    """
    Compute softmax over x
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def _beta_schedule(num_steps, start=1e-5, end=0.999, schedule='linear'):
    """Function to generate betas."""
    if schedule == 'linear':
        return torch.linspace(start, end, num_steps)
    elif schedule == 'cosine':
        betas = torch.linspace(np.pi / 2, 0, num_steps)
        betas = torch.cos(betas) * (end - start) + start
        return betas
    elif schedule == 'sohl-dickstein':
        betas = torch.linspace(0, num_steps-1, num_steps)
        betas = 1/(num_steps - betas + 1)
        return betas
    else:
        raise NotImplementedError(type)


def sample_transition_matrix(x_0, Q_bar):
    """
    Sample a markov transition according to next_step = x_0 * Q^time,
    where Q_bar = Q^t or cumprod of scheduled transition matrices
    returns sample and probabilities
    """
    p_next_step = torch.mm(x_0, Q_bar)
    next_step = torch.multinomial(p_next_step, num_samples=1)
    return next_step.squeeze(), p_next_step # sample and probabilities


class Tokenizer(object):
    def __init__(self, protein_alphabet=PROTEIN_ALPHABET, all_aas=MSA_AAS, gap=GAP):
        self.alphabet = list("".join(protein_alphabet))
        self.all_aas = list("".join(all_aas))
        self.gap = gap
        self.a_to_i = {u: i for i, u in enumerate(self.alphabet)}
        self.i_to_a = np.array(self.alphabet)
        self.K = len(self.all_aas)
    

    def q_uniform_schedule(self, num_steps=500, schedule='sohl-dickstein'):
        """
        Takes a number of time steps and a schedule and
        generates Q_t and Q_bar_t matrices.
        """
        betas = _beta_schedule(num_steps=num_steps, schedule=schedule)
        Q_t = []
        for i in range(len(betas)):
            # each iteration of the loop computes matrix Q_t at time t=i
            # using the equations from the paper for uniform transition matrix
            q_non_diag = torch.ones((self.K, self.K)) / self.K * betas[i]
            norm_constant = (1 - (q_non_diag).sum(axis=0))
            q_diag = torch.tensor(np.identity(self.K)) * norm_constant
            R = q_diag + q_non_diag
            Q_t.append(R)
        Q_prod = cumprod_matrix(Q_t)
        Q_prod = torch.stack(Q_prod)
        Q_t = torch.stack(Q_t)
        return Q_prod, Q_t
    
    def tokenize(self, seq):
        return np.array([self.a_to_i[a] for a in seq])
    
    def untokenize(self, x):
        if torch.is_tensor(x):
            return "".join([self.i_to_a[int(t.item())] for t in x])
        else:
            return "".join([self.i_to_a[t] for t in x])
    
    def one_hot(self, tokenized):
        "seq -> one hot"
        x_onehot = torch.nn.functional.one_hot(tokenized, num_classes=self.K)
        return x_onehot.to(torch.double)
    
    def undo_one_hot(self, x_onehot):
        "one hot -> seq"
        tokenized = [np.where(r==1)[0] for r in x_onehot]
        return tokenized


def custom_mutant_matrix(input_path, model, device, focus_seq, focus_seq_name, focus_seq_trimmed, tokenizer, 
                        loss_func1, loss_func2, _lambda, timesteps, Q, Q_bar, N_pred_iterations=10, minibatch_size=2000,
                        filename_prefix='', offset=0):
    start_idx, end_idx = focus_seq_name.split('/')[-1].split('-')
    start_idx = int(start_idx)

    wt_pos_focus_idx_tuple_list = []
    focus_seq_index = 0
    focus_seq_list = []
    mutant_to_letter_pos_idx_focus_list = {}

    for i, letter in enumerate(focus_seq):
        if letter == letter.upper():
            for mut in ALL_AAS:
                pos = start_idx+i
                if letter != mut:
                    mutant = letter+str(pos)+mut
                    mutant_to_letter_pos_idx_focus_list[mutant] = [letter, start_idx+i, focus_seq_index]
            focus_seq_index += 1
    
    mutant_sequences = ["".join(focus_seq_trimmed)]
    mutant_sequences_descriptor = ['wt']

    INPUT = open(input_path, 'r')
    for i, line in enumerate(INPUT):
        line = line.rstrip()
        if i >= 1:
            line_list = line.split(',')
            mutant_list = line_list[0].split(':')
            valid_mutant = True

            for mutant in mutant_list:
                if mutant not in mutant_to_letter_pos_idx_focus_list:
                    valid_mutant = False
            
            if valid_mutant:
                focus_seq_copy = list(focus_seq_trimmed)[:]

                for mutant in mutant_list:
                    wt_aa, pos, idx_focus = mutant_to_letter_pos_idx_focus_list[mutant]
                    mut_aa = mutant[-1]
                    focus_seq_copy[idx_focus] = mut_aa

                mutant_sequences.append("".join(focus_seq_copy))
                mutant_sequences_descriptor.append(":".join(mutant_list))
    INPUT.close()

    mutant_sequences = np.array(mutant_sequences)
    print(f'Number of mutations {mutant_sequences.shape[0] - 1}\n')
    prediction_matrix = np.zeros((mutant_sequences.shape[0], N_pred_iterations))
    batch_order = np.arange(mutant_sequences.shape[0])

    collater = CollaterV2(tokenizer=tokenizer, num_steps=timesteps, Q=Q, Q_bar=Q_bar)

    steps = timesteps

    model.eval()

    with torch.no_grad():
        for i in tqdm(range(N_pred_iterations), total=N_pred_iterations, desc='Iterations'):
            print(f'\nStarting iteration: {i+1}\n')
            np.random.shuffle(batch_order)

            # for j in range(0, len(mutant_sequences), minibatch_size):
            for j in tqdm(range(0, len(mutant_sequences), minibatch_size), total=len(mutant_sequences)//minibatch_size, desc='Batches'):
                
                batch_index = batch_order[j:j+minibatch_size]
                batch = mutant_sequences[batch_index]

                losses = torch.zeros(batch_index.shape[0]).to(device)

                for h in range(3):

                    if h == 0:
                        t = 1
                    elif h == 1:
                        t = np.random.randint(2, steps)
                    else:
                        t = steps

                    src, src_onehot, timesteps, tgt, tgt_onehot, Q, Q_bar, q = collater(batch, t)
                    
                    # move to data to device
                    q = q.to(device)
                    Q = Q.to(device)
                    Q_bar = Q_bar.to(device)
                    src_onehot = src_onehot.to(device)
                    tgt_onehot = tgt_onehot.to(device)
                    timesteps = timesteps.to(device)
                    src = src.to(device)
                    tgt = tgt.to(device)

                    # compute number of tokens in batch
                    n_tokens = src.shape[1]
                    
                    
                    if t == steps:
                        outputs = torch.zeros_like(src_onehot)
                    else:
                        outputs = model(src, timesteps)

                    lvb_loss = loss_func1(src_onehot, q, outputs, tgt, tgt_onehot, timesteps, Q, Q_bar)
                    lvb_loss = lvb_loss.to(torch.float32) # * n_tokens

                    if h == 1:
                        lvb_loss *= (steps - 1)

                    losses += lvb_loss

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix[idx_batch][i] = losses[k]

    
        mean_elbos = np.mean(prediction_matrix, axis=1).flatten().tolist()

        wt_elbo = mean_elbos.pop(0)
        mutant_sequences_descriptor.pop(0)

        delta_elbos = np.asarray(mean_elbos) - wt_elbo

        # flip sign because we computed the negative ELBO
        delta_elbos *= -1

        if filename_prefix == '':
            return mutant_sequences_descriptor, delta_elbos
        else:
            OUTPUT = open(filename_prefix+'_samples-'+str(N_pred_iterations)+'_elbo_predictions.csv', 'w')

            for i, descriptor in enumerate(mutant_sequences_descriptor):
                OUTPUT.write(descriptor+';'+str(delta_elbos[i])+'\n')
            
            OUTPUT.close()



class CollaterV2(object):
    """
    Collater for generating batch data according to markov process according to Austin et al.
    inputs:
        sequences : list of sequences
        tokenizer: Tokenizer()
        num_timesteps: number of diffusion timesteps

    outputs:
        src : source  masked sequences (model input)
        timesteps: (D-t+1) term
        tokenized: tokenized sequences (target seq)
        Q : markov matrix
        Q_bar : cumulative prod of markov matrix
        q_x : forward transition probabilities
    """
    def __init__(self, tokenizer=Tokenizer(), num_steps=500, Q=None, Q_bar=None):
        self.tokenizer=tokenizer
        self.num_steps = num_steps
        self.K = tokenizer.K
        self.Q = Q
        self.Q_bar = Q_bar
    
    def __call__(self, data, t):

        sequences = [x for x in data]

        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        one_hot = torch.stack([self.tokenizer.one_hot(seq) for seq in tokenized])

        src = []
        timesteps = []
        q_x = []
        src_one_hot = []

        for i, k in enumerate(tokenized):
            x = one_hot[i]
            timesteps.append(t)

            if t == self.num_steps:
                t = t - 1

            x_t, q_x_t = sample_transition_matrix(x, self.Q_bar[t])
            src.append(x_t)
            q_x.append(q_x_t)
            src_one_hot.append(self.tokenizer.one_hot(x_t))
        
        src = torch.stack(src).to(torch.long)
        src_one_hot = torch.stack(src_one_hot).to(torch.double)
        q_x = torch.stack(q_x).to(torch.double)
        timesteps = torch.tensor(timesteps, dtype=torch.long)
        tokenized = torch.stack(tokenized).to(torch.long)

        return src, src_one_hot, timesteps, tokenized, one_hot, self.Q, self.Q_bar, q_x

