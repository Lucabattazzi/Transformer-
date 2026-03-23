import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module): 
    # d_model = embedding dimension, vocab_size = size of the vocabulary
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # embedding layer provided by pytorch, from vocabulary index to embedding space

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model) # scale the embedding by the square root of the embedding dimension (as seen in the original transformer paper)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model) to hold the positional encodings
        pe = torch.zeros(seq_len, d_model)   # sequence length is the maximum length of the input sequence, and d_model is the embedding dimension

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # shape (seq_len, 1) (contains all possible positional indices)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # shape (d_model/2,) (contains the scaling factors for the sine and cosine functions). We use exp log for numerical stability
        # apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * denominator) 
        pe[:, 1::2] = torch.cos(position * denominator)

        pe = pe.unsqueeze(0) # shape (1, seq_len, d_model) (add a batch dimension in the first dimension, to account for more than one sentence)

        self.register_buffer('pe', pe) # the tensor is saved as the file is saved (it's not updated during training)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False) # add the positional encoding of each word to its embedding. This doesn't need to be trained.
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    # each batch is normalized indipendently. Epsilon is introduced for numerical stability
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # learnable multiplicative coefficients (to introduce more degrees of freedom)
        self.bias = nn.Parameter(torch.zeros(1)) # learnable bias term
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # calculate the mean of the input tensor along the last dimension (the embedding dimension)
        std = x.std(dim=-1, keepdim=True) # calculate the standard deviation of the input tensor along the last dimension (the embedding dimension)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias # normalize the input tensor and apply the learnable parameters
        
class FeedForwardBlock(nn.Module):
    # neural network with two layers and ReLu activation in between. The dimensions are d_model = embedding dimension, d_ff = dimension of the hidden layer (2048)
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # first linear layer
        self.dropout = nn.Dropout(dropout) # dropout layer for regularization
        self.linear_2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        # (Batch size, sequence length, embedding dimension) -> (Batch size, sequence length, d_ff) -> (Batch size, sequence length, embedding dimension)
        x = self.linear_1(x) # apply the first linear layer
        x = torch.relu(x) # apply the ReLU activation function
        x = self.dropout(x) # apply dropout for regularization
        x = self.linear_2(x) # apply the second linear layer
        return x
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None: # h indicates the number of heads
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model must be divisible by h" # ensure that the embedding dimension is divisible by the number of heads

        self.d_k = d_model // h # dimension of each head (the embedding dimension is divided by the number of heads)
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo

        self.dropout = nn.Dropout(dropout) # dropout layer for regularization

    @staticmethod # you don't have to have an instance of the class to call the method 

    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k= query.shape[-1]
        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len) 
        attention_scores = (query @ key.transpose(-2, -1))/ math.sqrt(d_k) # transpose(-2, -1) to swap the last two dimensions 
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9) # replace values with mask = 0 with very negative number
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores) 

        return(attention_scores @ value), attention_scores # latter used by visualization
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # shape (batch size, sequence length, d_model)
        key = self.w_k(k) # shape (batch size, sequence length, d_model)
        value = self.w_v(v) # shape (batch size, sequence length, d_model

        # splitting of query, key and value to perform multi-head attention
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # (Batch, Seq_Len, d_model) --> (Batch, Seq_LEn, h, d_k) -->(Batch, h, Seq_Len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) # (Batch, Seq_Len, d_model) --> (Batch, Seq_LEn, h, d_k) -->(Batch, h, Seq_Len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) # (Batch, Seq_Len, d_model) --> (Batch, Seq_LEn, h, d_k) -->(Batch, h, Seq_Len, d_k)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

    # conacatenations of the heads
    # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_model) 

        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) 
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) 
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # two residual connections, one for the self attention block and one for the feed forward block

    def forward(self, x, src_mask): # source mask to avoid interaction with padding words

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # self attention block with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
    
class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization() 

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask) # pass the input through each encoder block
        return self.norm(x) # apply layer normalization to the output of the last encoder block
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) # three residual connections, one for the self attention block, one for the cross attention block and one for the feed forward block

    def forward(self, x, encoder_output, src_mask, tgt_mask): # source comes from the encoder, target from the decoder

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # self attention block with residual connection
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # cross attention block with residual connection
        x = self.residual_connections[2](x, self.feed_forward_block) # feed forward block with residual connection
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization() 

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask) # pass the input through each decoder block
        return self.norm(x) 
    
class  ProjectionLayer(nn.Module): # from output of the decoder to the size of the vocabulary (logits for each word in the vocabulary)
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # linear layer to project the output of the decoder to the size of the vocabulary

    def forward(self, x):
        return self.proj(x) # shape (batch size, sequence length, vocab size) (logits for each word in the vocabulary)
        # softmax is already applied by the CrossEntropyLoss()
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed : InputEmbedding, tgt_embed: InputEmbedding, src_pos:PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed= tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src) 
        src = self.src_pos(src) 
        return self.encoder(src, src_mask) 
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt) 
        tgt = self.tgt_pos(tgt) 
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff = 2048) -> Transformer:

    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):  
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the weights with Xavier
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
