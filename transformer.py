import json
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import collections
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")


hr_flat_tokens = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        '[', '\\', ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 
                        '{', '|', '}', '~', 'č', 'ć', 'đ', 'š', 'ž']
en_flat_tokens = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        '[', '\\', ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 
                        '{', '|', '}', '~']

# Count character frequencies
hr_vocab_counter = collections.Counter(hr_flat_tokens)
en_vocab_counter = collections.Counter(en_flat_tokens)

# Special tokens and vocabulary
special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]
hr_vocab = special_tokens + sorted(hr_vocab_counter.keys())
en_vocab = special_tokens + sorted(en_vocab_counter.keys())

# Create token-to-index and index-to-token dictionaries
hr_token2index = {token: idx for idx, token in enumerate(hr_vocab)}
en_token2index = {token: idx for idx, token in enumerate(en_vocab)}
hr_index2token = {idx: token for idx, token in enumerate(hr_vocab)}
en_index2token = {idx: token for idx, token in enumerate(en_vocab)}

def sentence2index(src, sentence_tokens, sentence_length, start_token=False, end_token=False):
    token2index = hr_token2index if src == "hr" else en_token2index

    if start_token:
        sentence_index = [token2index["<START>"]]
    else:
        sentence_index = []

    sentence_index += [token2index[token.lower()] if token.lower() in token2index else token2index["<UNK>"] for token in list(sentence_tokens)]

    if end_token:
        sentence_index.append(token2index["<END>"])

    sentence_index += [token2index["<PAD>"] for _ in range(sentence_length - len(sentence_index))]

    return sentence_index

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # matmul Q and K and scale
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax layer
        attention = torch.softmax(scores, dim=-1)

        # matmul attention and V
        context = torch.matmul(attention, V)

        return context
    
    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()

        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()

        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    def forward(self, Q, K, V, mask=None):
        # batch_size x seq_len x d_model
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # batch_size x num_heads x seq_len x d_k
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention = self.scaled_dot_product_attention(Q, K, V, mask)

        # batch_size x seq_len x d_model
        attention = self.combine_heads(attention)

        output = self.output_linear(attention)

        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x) # d_model x d_ff
        x = torch.relu(x)
        x = self.linear2(x) # d_ff x d_model

        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # multi-head attention
        attention = self.multi_head_attention(x, x, x, mask)

        # add and norm
        x = self.layer_norm1(x + attention)
        x = self.dropout1(x)

        # position-wise feed forward
        feed_forward = self.position_wise_feed_forward(x)

        # add and norm
        x = self.layer_norm2(x + feed_forward)
        x = self.dropout2(x)

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.multi_head_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention2 = MultiHeadAttention(d_model, num_heads)

        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, trg_mask=None):
        # masked multi-head attention
        masked_attention = self.multi_head_attention1(x, x, x, trg_mask)

        # add and norm
        x = self.layer_norm1(x + masked_attention)
        x = self.dropout1(x)

        # multi-head attention
        attention = self.multi_head_attention2(x, encoder_output, encoder_output, src_mask)

        # add and norm
        x = self.layer_norm2(x + attention)
        x = self.dropout2(x)

        # position-wise feed forward
        feed_forward = self.position_wise_feed_forward(x)

        # add and norm
        x = self.layer_norm3(x + feed_forward)
        x = self.dropout3(x)

        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        
        output = self.linear(tgt)

        return output
    
def load_transformer(model_id):
    with open(f"./models/{model_id}.json", "r") as file:
        model_args = json.load(file)

    transformer = Transformer(**model_args)
    transformer.load_state_dict(torch.load(f"./models/{model_id}.pt"))
    transformer = transformer.to(device)
    transformer.eval()

    return transformer

def translate(sentance, model, max_seq_length=60):
    result = ""
    src = torch.tensor([sentence2index("hr", sentance, max_seq_length)]).to(device)
    
    for i in range(60):
        tgt = torch.tensor([sentence2index("en", result, max_seq_length, start_token = True)]).to(device)

        output = model(src, tgt)

        next_word_prob = output[0][i]
        next_word_index = torch.argmax(next_word_prob).item()
        next_word = en_index2token[next_word_index]

        if next_word == "<END>":
            break

        result += next_word

    return result