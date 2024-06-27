import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from sequence_models.layers import PositionFeedForward, DoubleEmbedding
from sequence_models.convolutional import ByteNetBlock
from constants import PROTEIN_ALPHABET, MASK, MSA_PAD
from esm.modules import TransformerLayer, LearnedPositionalEmbedding, RobertaLMHead, ESM1bLayerNorm, AxialTransformerLayer


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        Used for encoding timestep in diffusion models

        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) * -(np.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        device = x.device
        pe = pe.to(device)
        return pe[x] # .to(x.device)

class PositionalEncoding(nn.Module):

    """
    2D Positional encoding for transformer
    :param d_model: dimension of the model
    :param max_len: max number of positions
    """

    def __init__(self, d_model, max_len=2048):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x.reshape(x.shape[1], x.shape[0], x.shape[2]) # [b x l x e]

class ByteNetTime(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu', down_embed=True,
                 timesteps=None):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        :param timesteps: None or int providing max timesteps in DM model
        """
        super().__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_embedding, timesteps) # Timestep encoding
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
            else:
                self.embedder = DoubleEmbedding(n_tokens - n_frozen_embs, n_frozen_embs,
                                                d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()
        if down_embed:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert n_tokens == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, y, input_mask=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x, y, timesteps=self.timesteps)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x, y, timesteps=None):
        e = self.embedder(x)
        if timesteps is not None:
            e2 = self.time_encoding(y)
            # expand dim of e2 to match e1
            e2 = e2.expand(e.shape[1], e2.shape[0], e2.shape[1])
            e2 = e2.reshape(e.shape[0], e.shape[1], e.shape[2])
            e = torch.add(e2, e)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class ByteNetLMTime(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, final_ln=False, slim=True, activation='relu',
                 tie_weights=False, down_embed=True, timesteps=None):
        super().__init__()
        self.embedder = ByteNetTime(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
            self.decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

    def forward(self, x, y, input_mask=None):
        e = self.embedder(x, y, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)


class MSATransformerTime(nn.Module):
    """
    Based on implementation described by Rao et al. in "MSA Transformer"
    https://doi.org/10.1101/2021.02.12.430858
    Args:
        d_model: int,
            embedding dimension of model
        d_hidden: int,
            embedding dimension of feed forward network
       n_layers: int,
           number of layers
       n_heads: int,
           number of attention heads
   """

    def __init__(self, d_model, d_hidden, n_layers, n_heads, use_ckpt=False, n_tokens=len(PROTEIN_ALPHABET),
                 padding_idx=PROTEIN_ALPHABET.index(MSA_PAD), mask_idx=PROTEIN_ALPHABET.index(MASK),
                 max_positions=1024, timesteps=None):
        super(MSATransformerTime, self).__init__()

        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_model, timesteps) # Timestep encoding
        self.embed_tokens = nn.Embedding(
            n_tokens, d_model, padding_idx=mask_idx
        )
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    d_model, d_hidden, n_heads
                )
                for _ in range(n_layers)
            ]
        )
        self.padding_idx = padding_idx

        # self.contact_head = ContactPredictionHead()
        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, padding_idx)
        self.emb_layer_norm_before = nn.LayerNorm(d_model)
        self.emb_layer_norm_after = nn.LayerNorm(d_model)
        self.lm_head = RobertaLMHead(
            embed_dim=d_model,
            output_dim=n_tokens,
            weight=self.embed_tokens.weight
        )

        self.use_ckpt = use_ckpt

    def forward(self, tokens, timesteps):
        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C
        # print("tokens", tokens.shape) # B, D, L (batch, depth length)
        x = self.embed_tokens(tokens)
        x = x + self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(x.size())
        x = self.emb_layer_norm_before(x)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        # print("x", x.shape) # B, D, L, E

        y = self.time_encoding(timesteps)
        y = y.unsqueeze(1).unsqueeze(1)
        y = y.expand(y.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x += y

        # ADD 1 to query sequence in MSA (encode query sequence)
        q = torch.zeros(x.shape)
        q = q.to(x.device)
        q[:,0,:,0] += 1 # add encoding to 1st sequence (query seq) in MSA
        x += q
        #

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = checkpoint(layer, x, None, padding_mask, False)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        x = self.lm_head(x)
        return x


class DiffusionEncoder(nn.Module):
    def __init__(self, device, params):
        super(DiffusionEncoder, self).__init__()
        self.device = device
        self.seq_len = params['seq_len']
        self.n_tokens = params['n_tokens']
        self.timesteps = params['timesteps']
        self.embedding_dim = params['embedding_dim']
        self.hidden_layers_sizes = params['hidden_layers_sizes']
        self.z_dim = params['z_dim']
        self.convolve_input = params['convolve_input']
        self.convolution_depth = params['convolution_depth']
        self.dropout_proba = params['dropout_proba']


        self.layer_bias_init = 0.1

        self.time_encoding = PositionalEncoding1D(self.embedding_dim, self.timesteps)
        self.embedder = nn.Embedding(self.n_tokens, self.embedding_dim)

        if self.convolve_input:
            self.input_convolution = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.convolution_depth,
                                               kernel_size=1, stride=1, bias=False)
            self.channel_size = self.convolution_depth
        else:
            self.channel_size = self.embedding_dim
        
        self.hidden_layers = torch.nn.ModuleDict()
        for layer_index in range(len(self.hidden_layers_sizes)):
            if layer_index == 0:
                self.hidden_layers[str(layer_index)] = nn.Linear((self.channel_size * self.seq_len), 
                                                                 self.hidden_layers_sizes[layer_index])
            else:
                self.hidden_layers[str(layer_index)] = nn.Linear(self.hidden_layers_sizes[layer_index - 1],
                                                                 self.hidden_layers_sizes[layer_index])
        
        self.fc_out = nn.Linear(self.hidden_layers_sizes[-1], self.z_dim)
        nn.init.constant_(self.fc_out.bias, self.layer_bias_init)

        if params['activation'] == 'relu':
            self.nonlinear_activation = nn.ReLU()
        elif params['activation'] == 'tanh':
            self.nonlinear_activation = nn.Tanh()
        elif params['activation'] == 'sigmoid':
            self.nonlinear_activation = nn.Sigmoid()
        elif params['activation'] == 'elu':
            self.nonlinear_activation = nn.ELU()
        elif params['activation'] == 'linear':
            self.nonlinear_activation = nn.Identity()
        
        if self.dropout_proba > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_proba)
    
    def forward(self, x, t):
        x = self.embedder(x)
        t_encoded = self.time_encoding(t)
        t_encoded = t_encoded.expand(x.shape[1], t_encoded.shape[0], t_encoded.shape[1])
        t_encoded = t_encoded.reshape(x.shape[0], x.shape[1], x.shape[2])
        x = x + t_encoded

        if self.convolve_input:
            x = x.permute(0, 2, 1)
            x = self.input_convolution(x)
            x = x.view(-1, self.seq_len * self.channel_size)
        else:
            x = x.view(-1, self.seq_len * self.channel_size)
        

        for layer_index in range(len(self.hidden_layers_sizes)):
            x = self.nonlinear_activation(self.hidden_layers[str(layer_index)](x))
            if self.dropout_proba > 0.0:
                x = self.dropout_layer(x)
        
        x = self.fc_out(x)
        return x



class DiffusionDecoder(nn.Module):
    def __init__(self, device, params):
        super(DiffusionDecoder, self).__init__()
        self.device = device
        self.seq_len = params['seq_len']
        self.n_tokens = params['n_tokens']
        self.hidden_layers_sizes = params['hidden_layers_sizes']
        self.z_dim = params['z_dim']
        self.convolve_output = params['convolve_output']
        self.convolution_depth = params['convolution_depth']
        self.dropout_proba = params['dropout_proba']
        self.include_temperature_scaler = params['include_temperature_scaler']
        self.include_sparsity = params['include_sparsity']
        self.num_tiles_sparsity = params['num_tiles_sparsity']


        self.layer_bias_init = 0.1

        # hidden layers
        self.hidden_layers = nn.ModuleDict()
        for layer_index in range(len(self.hidden_layers_sizes)):
            if layer_index == 0:
                self.hidden_layers[str(layer_index)] = nn.Linear(self.z_dim, self.hidden_layers_sizes[layer_index])
            else:
                self.hidden_layers[str(layer_index)] = nn.Linear(self.hidden_layers_sizes[layer_index - 1], self.hidden_layers_sizes[layer_index])
            nn.init.constant_(self.hidden_layers[str(layer_index)].bias, self.layer_bias_init)
        
        # Non-linearities
        activation_funcs = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'linear': nn.Identity()
        }
        self.first_hidden_nonlinearity = activation_funcs[params['first_hidden_nonlinearity']]
        self.last_hidden_nonlinearity = activation_funcs[params['last_hidden_nonlinearity']]

        if self.dropout_proba > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_proba)
        
        if self.convolve_output:
            self.output_convolution = nn.Conv1d(in_channels=self.convolution_depth, out_channels=self.n_tokens,
                                                kernel_size=1, stride=1, bias=False)
            self.channel_size = self.convolution_depth
        else:
            self.channel_size = self.n_tokens
        
        if self.include_sparsity:
            self.sparsity_weight = nn.Parameter(torch.randn(int(self.hidden_layers_sizes[-1] / self.num_tiles_sparsity), self.seq_len))
        
        self.W_out = nn.Parameter(torch.zeros(self.channel_size * self.seq_len, self.hidden_layers_sizes[-1]))
        nn.init.xavier_normal_(self.W_out)
        self.b_out = nn.Parameter(torch.zeros(self.n_tokens * self.seq_len))
        nn.init.constant_(self.b_out, self.layer_bias_init)

        if self.include_temperature_scaler:
            self.temperature_scaler = nn.Parameter(torch.ones(1))

    def forward(self, x):
        batch_size = x.shape[0]

        if self.dropout_proba > 0.0:
            x = self.dropout_layer(x)
        
        for layer_index in range(len(self.hidden_layers_sizes) - 1):
            x = self.first_hidden_nonlinearity(self.hidden_layers[str(layer_index)](x))
            if self.dropout_proba > 0.0:
                x = self.dropout_layer(x)
        
        x = self.last_hidden_nonlinearity(self.hidden_layers[str(len(self.hidden_layers_sizes)-1)](x))
        if self.dropout_proba > 0.0:
            x = self.dropout_layer(x)
        
        W_out = self.W_out.data

        if self.convolve_output:
            W_out = torch.mm(W_out.view(self.seq_len * self.hidden_layers_sizes[-1], self.channel_size),
                             self.output_convolution.weight.view(self.channel_size, self.n_tokens))
            
        if self.include_sparsity:
            sparsity_tiled = self.sparsity_weight.repeat(self.num_tiles_sparsity, 1)
            sparsity_tiled = torch.sigmoid(sparsity_tiled).unsqueeze(2)
            W_out = W_out.view(self.hidden_layers_sizes[-1], self.seq_len, self.n_tokens) * sparsity_tiled
        
        W_out = W_out.view(self.seq_len * self.n_tokens, self.hidden_layers_sizes[-1])

        x = F.linear(x, weight=W_out, bias=self.b_out)

        if self.include_temperature_scaler:
            x = torch.log(1.0 + torch.exp(self.temperature_scaler)) * x
        
        x = x.view(batch_size, self.seq_len, self.n_tokens)
        return x


class DenseDiffusionModel(nn.Module):
    def __init__(self, device, encoder_params, decoder_params):
        super().__init__()

        self.encoder = DiffusionEncoder(device=device, params=encoder_params)
        self.decoder = DiffusionDecoder(device=device, params=decoder_params)

    def forward(self, x, t):
        x = self.encoder(x, t)
        x = self.decoder(x)
        return x

