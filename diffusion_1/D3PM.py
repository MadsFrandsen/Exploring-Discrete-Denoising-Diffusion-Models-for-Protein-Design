import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import einops
import matplotlib.pyplot as plt

from tqdm import tqdm

from diffusion_1.model2 import ContextUnet
# from dataset_discretize import DiscretizeD3PMNIST


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)



class D3PMTrainer(pl.LightningModule):
    """
    Trainer module for the MNIST model.
    """

    def __init__(self, hidden_dim=128, num_bins=2, time_steps=1000):
        """
        Args:
            hidden_dim (int): hidden dimension of the model
            num_bins (int): number of bins to discretize the data into
            nb_block (int): number of blocks in the model
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.time_steps = time_steps

        self.model = ContextUnet(in_channels=1, n_feat=hidden_dim, n_cfeat=10, img_size=28, nb_class=num_bins)
        self.loss = nn.CrossEntropyLoss()

        self.apply(init_weights)
    
    def forward(self, data, t):
        data = data.unsqueeze(1)
        t = t.unsqueeze(1)

        logits = self.model(data, t)

        return logits
    
    def compute_loss(self, logits, data, init_data):
        loss_vb = self.loss(logits, data)
        loss_init = 0.0 * self.loss(logits, init_data)

        loss = loss_vb + loss_init
        return loss, (loss_vb, loss_init)
    
    def training_step(self, batch, _):
        data, data_next, _, init_data, time_step = batch

        logits = self.forward(data_next.long(), time_step.float())

        loss, (loss_vb, loss_init) = self.compute_loss(
            logits, data.long(), init_data.long()
        )

        self.log("train_loss", loss)
        self.log("train_loss_vb", loss_vb)
        self.log("train_loss_init", loss_init)

        return loss
    
    def on_train_epoch_end(self):
        self.eval()
        with torch.no_grad():
            self.generate()
    
    def generate(self):
        device = self.device
        self.eval()

        data = torch.randint(0, self.num_bins, (1, 28, 28)).long().to(device)
        
        time_step = torch.tensor([1.0]).to(device)
        
        plot_index = [0, 1, 50, 100, 150, 200, 240]

        for i in range(self.time_steps):
            if i in plot_index:
                self.save_image(data, i)
            
            if data.shape[1] == 1:
                data = data.squeeze(1)
            
            logits = self.forward(data, time_step)
            logits = logits.permute(0, 2, 3, 1)
            logits_flatten = einops.rearrange(logits, "a b c d -> (a b c) d")

            proba = F.softmax(logits_flatten, dim=1)

            data = torch.distributions.Categorical(probs=proba).sample()

            data = einops.rearrange(
                data,
                "(a b c) -> a b c",
                a=logits.shape[0],
                b=logits.shape[1],
                c=logits.shape[2],
            )

            time_step = time_step - 1.0 / self.time_steps

        self.save_image(data, self.time_steps)
        self.train()
    
    def save_image(self, data, i):
        # plot the data
        plt.imshow(data.squeeze().cpu().numpy(), cmap="gray")

        # title
        plt.title(f"data = {i}")

        # save the figure
        plt.savefig(f"/Users/madsfrandsen/Documents/BSc_project/images/data_{i}.png")

        # close the figure
        plt.close()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer







# class D3PMTrainer(nn.Module):
#     def __init__(self, device, model: ContextUnet, T, num_bins):
#         super(D3PM, self).__init__()

#         self.device = device
#         self.model = model
#         self.time_steps = T
#         self.num_bins = num_bins

#         self.beta_t = generate_betas(T)
        
#         self.transition_matrices = compute_transition_matrix(
#             self.beta_t, T, num_bins
#         )

#         self.cumulated_transition = compute_acc_transition_matrices(
#             self.transition_matrices
#         )
    
#     def forward(self, data, t):
#         data = data.unsqueeze(1)
#         t = t.unsqueeze(1)

#         logits = self.model(data, t)

#         return logits
    

    


    





