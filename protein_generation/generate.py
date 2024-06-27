import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import Tokenizer
from dataset import ProteinSequenceDatasetMSA, CollaterMSA


def generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len, device, batch_size=3):
    """
    Generate a random start string from uniform dist and convert to predictions
    """

    model.eval()
    
    sample = torch.randint(0, tokenizer.K, (batch_size, seq_len))
    sample = sample.to(torch.long)
    sample = sample.to(device)
    Q = Q.to(device)
    Q_bar = Q_bar.to(device)

    # iterate over reverse timesteps
    timesteps = torch.linspace(timesteps-1, 1, int((timesteps-1)/1), dtype=int)
    timesteps = timesteps.to(device)
    with torch.no_grad():
        for t in timesteps:
            timesteps = torch.tensor([t] * batch_size).to(device)
            predictions = model(sample, timesteps)
            p = predictions[:, :, :tokenizer.K]
            p = torch.nn.functional.softmax(p, dim=-1) # softmax over categorical probs
            p = p.to(torch.float64)
            x_tminus1 = sample.clone()
            for i, s in enumerate(sample):
                x_t_b = tokenizer.one_hot(s)
                A = torch.mm(x_t_b, torch.t(Q[t])) # [P x K]
                Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K) # [P x K x K]
                B_pred = torch.mul(p[i].unsqueeze(2), Q_expand)
                q_t = torch.mul(A.unsqueeze(1), B_pred) # [P x K x K]
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1, 2), p[i].unsqueeze(2)).squeeze()
                p_theta_marg = p_theta_marg / p_theta_marg.sum(dim=1, keepdim=True)
                x_tminus1[i] = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
            sample = x_tminus1

        untokenized = [tokenizer.untokenize(s) for s in sample]
        return sample, untokenized



def generate_msa_d3pm(model, batch_size, n_sequences, seq_length, device, timesteps=500, 
                      penalty_value=0, Q_bar=None, Q=None, tokenizer=Tokenizer(), start_query=True):
    
    model.eval()
    
    sample = torch.randint(0, tokenizer.K, (batch_size, n_sequences, seq_length))
    if start_query:
        query_sequence = get_ref_seq(tokenizer, Q_bar, Q, timesteps)
        for i in range(batch_size):
            sample[i][0] = query_sequence[0]

    sample = sample.to(torch.long)
    sample = sample.to(device)
    Q = Q.to(device)
    Q_bar = Q_bar.to(device)

    [print("input query seq", tokenizer.untokenize(sample[i].flatten()[:seq_length])) for i in range(batch_size)]

    timesteps = torch.linspace(timesteps-1, 1, int((timesteps-1)/1), dtype=int)
    with torch.no_grad():
        for t in timesteps:
            timesteps = torch.tensor([t] * batch_size).to(device)
            predictions = model(sample, timesteps)
            p = predictions[:, :, :, :tokenizer.K]
            p = torch.nn.functional.softmax(p, dim=-1)
            p = p.to(torch.float64)
            x_tminus1 = sample.clone()
            for i, s in enumerate(sample):
                p_current = p[i].flatten(start_dim=0, end_dim=1)
                x_t_b = torch.stack([tokenizer.one_hot(s_i) for s_i in s])
                x_t_b = x_t_b.flatten(start_dim=0, end_dim=1)
                A = torch.mm(x_t_b, torch.t(Q[t])) # [P x K]
                Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K) # [P x K x K]
                B_pred = torch.mul(p_current.unsqueeze(2), Q_expand)
                q_t = torch.mul(A.unsqueeze(1), B_pred)
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1, 2), p_current.unsqueeze(2)).squeeze()
                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                penalty = torch.ones(p_theta_marg.shape).to(p_theta_marg.device)
                penalty[:, -1] += penalty_value
                p_theta_marg /= penalty
                x_tminus1_temp = torch.multinomial(p_theta_marg[:, :], num_samples=1).squeeze()
                x_tminus1_temp[:seq_length] = torch.multinomial(p_theta_marg[:seq_length, :-1], num_samples=1).squeeze()
                if start_query:
                    x_tminus1[i, 1:, :] = x_tminus1_temp.reshape(-1, seq_length)[1:, :]
                else:    
                    x_tminus1[i] = x_tminus1_temp.reshape(n_sequences, seq_length)
                sample = x_tminus1

                if t % 50 == 0:
                    print("time", t, tokenizer.untokenize(sample[0].flatten()[seq_length:seq_length*5]))
    untokenized = [[tokenizer.untokenize(sample[i].flatten())] for i in range(batch_size)]
    return sample, untokenized


def get_ref_seq(tokenizer, Q_bar, Q, timesteps):
    dataset = ProteinSequenceDatasetMSA(full_data=True, train=True, num_samples=1)
    collater = CollaterMSA(tokenizer=tokenizer, num_steps=timesteps, Q=Q, Q_bar=Q_bar)
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collater)

    query_msa = []

    for batch in dataloader:
        src, src_one_hot, timestep, tgt, tgt_one_hot, Q, Q_prod, q = batch
        query_msa.append(tgt[0][0])
        break

    return query_msa



def sample_and_save(out_path, num_seqs, model, tokenizer, Q, Q_bar, timesteps, seq_len, device, batch_size=3):
    string = []
    sample = []

    for i in range(num_seqs):
        i_sample, i_string = generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len, device, batch_size=1)
        string.append(i_string)
        sample.append(i_sample)
        print("generated sample: ", i)
    
    with open(out_path + 'generated_samples_string.fasta', 'w') as f:
        for i, _s, in enumerate(string):
            f.write('>SEQUENCE_' + str(i) + '\n' + str(_s[0]) + '\n')


def sample_and_save_msa(out_path, num_seqs, model, tokenizer, Q, Q_bar, timesteps, seq_len, device, batch_size,
                        penalty_value):
    sample, string = generate_msa_d3pm(model, batch_size, num_seqs, seq_len, device, timesteps, penalty_value, 
                                       Q_bar, Q, tokenizer)
    
    for count, msa in enumerate(string):
        fasta_string = ""
        with open(out_path + 'generated_msas.a3m', 'a') as f:
            for seq in range(num_seqs):
                seq_num = seq * seq_len
                next_seq_num = (seq+1) * seq_len
                seq_string = str(msa[0][seq_num:next_seq_num]).replace('!', '')
                if seq_num == 0:
                    f.write(">MSA_0" + "\n" + str(seq_string) + "\n")
                else:
                    f.write(">tr \n" + str(seq_string) + "\n")
            f.write(fasta_string)
            f.close()
