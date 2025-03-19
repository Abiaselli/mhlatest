import torch
import torch.nn as nn
import torch.fft
import logging
import math
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys
from transformers import PreTrainedTokenizerFast
import re
import torch.utils.checkpoint as checkpoint
import random
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

########################################
# Tokenizer
########################################
class HierarchicalTokenizer:
    def __init__(self, base_tokenizer, chunk_size=30):
        """
        Implements hierarchical tokenization by breaking input into fixed-size chunks 
        and mapping them to higher levels.
        """
        self.base_tokenizer = base_tokenizer
        self.tokenizer = base_tokenizer
        self.chunk_size = chunk_size
        self.hierarchy_vocab = {}  # Maps hierarchical chunks to unique token IDs
        self.next_hierarchical_id = base_tokenizer.vocab_size  # Start numbering beyond vocab
        self.pad_token_id = base_tokenizer.pad_token_id  # ✅ Ensure it's stored as a single integer
        # ✅ If `pad_token_id` is None, define one manually
        if self.pad_token_id is None:
            self.pad_token_id = base_tokenizer.eos_token_id  # Use EOS token as padding (or define custom ID)
            if self.pad_token_id is None:
                self.pad_token_id = 0  # As a last resort, set pad token to 0
        self.eos_token_id = base_tokenizer.eos_token_id  # ✅ Ensure it's stored as a single integer
        self.bos_token_id = base_tokenizer.bos_token_id  # ✅ Ensure it's stored as a single integer
        self.unk_token_id = base_tokenizer.unk_token_id  # ✅ Ensure it's stored as a single integer


    def __len__(self):
        """ Returns the updated vocabulary size including new hierarchical tokens. """
        return self.next_hierarchical_id  # Base vocab size + dynamically added tokens
    @property
    def vocab_size(self):
        """Returns the total vocabulary size, ensuring it exists."""
        #print(f"🔍 DEBUG: tokenizer type = {type(self.tokenizer)}")
        #print(f"🔍 DEBUG: tokenizer attributes = {dir(self.tokenizer)}")

        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size  # Use base tokenizer vocab size if available
        
        if hasattr(self, 'levels') and isinstance(self.levels, dict): 
            return max(len(vocab) for vocab in self.levels.values())  # Use hierarchical vocab sizes
        
        raise AttributeError("🚨 HierarchicalTokenizer has no valid vocabulary source!")

    def tokenize(self, text):
        """ Tokenizes text into hierarchical chunks. """
        base_tokens = self.base_tokenizer.encode(text)

        # 🔹 Debug: Check if base tokenizer is working
        if not base_tokens:
            print(f"⚠️ WARNING: Base tokenizer returned empty tokens for text: {text}")

        chunked_tokens = [base_tokens[i:i + self.chunk_size] for i in range(0, len(base_tokens), self.chunk_size)]

        hierarchical_tokens = []
        for chunk in chunked_tokens:
            chunk_tuple = tuple(chunk)  # Convert to immutable type for dictionary lookup
            if chunk_tuple not in self.hierarchy_vocab:
                self.hierarchy_vocab[chunk_tuple] = self.next_hierarchical_id
                self.next_hierarchical_id += 1  # Assign new ID
            hierarchical_tokens.append(self.hierarchy_vocab[chunk_tuple])

        # 🔹 Debug: Check if hierarchical tokens are being generated
        if not hierarchical_tokens:
            print(f"⚠️ WARNING: Hierarchical tokenization failed for text: {text}")

        return hierarchical_tokens if hierarchical_tokens else [self.base_tokenizer.pad_token_id]


    def decode(self, hierarchical_tokens):
        """ Decodes hierarchical tokens back into text using chunk lookup. """
        decoded_text = []
        for token in hierarchical_tokens:
            for chunk, token_id in self.hierarchy_vocab.items():
                if token == token_id:
                    decoded_text.append(self.base_tokenizer.decode(list(chunk)))
                    break
        return " ".join(decoded_text)


########################################
# 1. Build a Byte-Level Tokenizer/Vocab
########################################
seq_len = 500

from transformers import AutoTokenizer

# Load the pretrained tokenizer
base_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Wrap it with the hierarchical tokenizer
tokenizer = HierarchicalTokenizer(base_tokenizer)


########################################
# 2. Data Extraction
########################################

def extract_data(json_data):
    """Extracts training data from JSON file and tokenizes it."""
    input_ids_list = []
    target_ids_list = []

    for item in json_data:
        conversations = item.get("conversations", [])

        if not isinstance(conversations, list) or len(conversations) < 2:
            print(f"⚠️ Skipping entry with no valid conversation: {item}")
            continue

        for i in range(len(conversations) - 1):
            user_turn = conversations[i]
            assistant_turn = conversations[i + 1]

            # Ensure we only process valid user-assistant exchanges
            if user_turn.get("from") in ["user", "human"] and assistant_turn.get("from") in ["assistant", "gpt"]:
                query = user_turn.get("value", "").strip()
                target = assistant_turn.get("value", "").strip()

                # 🔹 Ensure valid text exists before tokenizing
                if not query or not target:
                    print(f"⚠️ Skipping empty user/assistant exchange: {user_turn} -> {assistant_turn}")
                    continue  

                input_ids = tokenizer.tokenize(query)
                target_ids = tokenizer.tokenize(target)

                # 🔹 Ensure tokenized output isn't empty
                if not input_ids or not target_ids:
                    print(f"⚠️ Skipping invalid tokenized entry: {query} -> {input_ids}")
                    continue

                input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
                target_ids_list.append(torch.tensor(target_ids, dtype=torch.long))

    return list(zip(input_ids_list, target_ids_list))  # Ensure format is (input, target)


########################################
# 3. Dataset and Collate Function
########################################

class ChatDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_seq_length):
        """Initialize dataset and tokenize the data properly."""
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # 🔹 Ensure data is correctly processed
        self.data = extract_data(json_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns exactly two elements: (input, target)."""
        return self.data[idx]

def collate_fn(batch, max_length, tokenizer):
    src_batch, tgt_batch = zip(*batch)

    pad_token_id = tokenizer.pad_token_id or 0  # Ensure pad token is valid

    # Convert to tensors and clamp token IDs within valid vocab range
    src_batch = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in src_batch]
    tgt_batch = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in tgt_batch]

    # ✅ Sanity Check: Detect any invalid token IDs
    for i, seq in enumerate(src_batch):
        
        if (seq < 0).any() or (seq >= tokenizer.vocab_size).any():
            print(f"🚨 Invalid token in source sequence {i}: {seq.tolist()}")

    for i, seq in enumerate(tgt_batch):
        if (seq < 0).any() or (seq >= tokenizer.vocab_size).any():
            print(f"🚨 Invalid token in target sequence {i}: {seq.tolist()}")

    # Pad sequences to max_length
    src_batch = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in src_batch]
    tgt_batch = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in tgt_batch]

    return torch.stack(src_batch), torch.stack(tgt_batch)


##############################################
# Positional Encoding (Standard Sin/Cos Version)
##############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Instead of erroring, simply truncate positional encodings to x.size(1)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
########################################
#Base Model
########################################


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent):
        """
        Multi-Head Latent Attention (MHLA)
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_latent: Compressed latent space dimension
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)  # Compress keys/values
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct keys
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct values

    def forward(self, x, memory=None):
        """
        Forward pass with optional memory (for hierarchical tokenization)
        - x: Input tensor (batch, seq_len, d_model)
        - memory: Cached latent state (batch, d_latent) [optional]
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)  # Merge and compress
        if memory is not None:
            latent_kv = (latent_kv + memory) / 2  # Combine with previous memory

        # Reconstruct full-size keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Multi-head split
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.W_o(attn_output)

        return output, latent_kv  # Return output and memory for next layer


class TimeAwareMultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, lambda_decay=0.01):
        """
        Multi-Head Latent Attention (MHLA) with Time-Aware Decay.
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_latent: Compressed latent space dimension
        - lambda_decay: Controls how quickly attention fades over time
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.lambda_decay = lambda_decay

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)  # Compress keys/values
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct keys
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct values

    def forward(self, x, memory=None):
        """
        Forward pass with optional hierarchical memory.
        - x: Input tensor (batch, seq_len, d_model)
        - memory: Cached latent state (batch, d_latent) [optional]
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)  # Merge and compress
        if memory is not None:
            latent_kv = (latent_kv + memory) / 2  # Combine with previous memory

        # Reconstruct full-size keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Multi-head split
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Compute raw attention scores
        attn_scores = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / math.sqrt(self.d_model)

        # 🔹 Apply time decay to attention scores
        time_matrix = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(seq_len, seq_len)
        time_decay = torch.exp(-self.lambda_decay * torch.abs(time_matrix - time_matrix.T))  # e^(-λt)
        attn_scores = attn_scores * time_decay.unsqueeze(0).unsqueeze(0)  # Shape: (batch, heads, seq, seq)

        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.W_o(attn_output)

        return output, latent_kv  # Return output and hierarchical memory

class HierarchicalMultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, lambda_decay=0.01, memory_size=5):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.lambda_decay = lambda_decay
        self.memory_size = memory_size  # How many past summaries to retain
        self.memory = []  # Stores hierarchical memory embeddings

        # Ensure `d_model` is evenly divisible by `num_heads`
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.head_dim = d_model // num_heads  # Compute per-head dimension

        # Standard attention components
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)

        # 🔹 Fix: Ensure Latent Memory Doesn't Accumulate Unexpectedly
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # Keep memory size consistent
        self.memory.append(latent_kv.mean(dim=1))  # Store compressed memory state

        # Reconstruct keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # 🔹 Fix: Ensure Shape Matches Expected Multi-Head Attention Shape
        try:
            k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        except RuntimeError as e:
            print(f"Error reshaping k/v in MHLA: {e}")
            print(f"Shape mismatch: batch={batch_size}, seq_len={seq_len}, num_heads={self.num_heads}, head_dim={self.head_dim}")
            raise e

        # Compute attention
        attn_scores = torch.matmul(q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), 
                                   k_reconstructed.transpose(-2, -1)) / math.sqrt(self.d_model)

        # Apply time decay
        time_matrix = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(seq_len, seq_len)
        time_decay = torch.exp(-self.lambda_decay * torch.abs(time_matrix - time_matrix.T))
        attn_scores = attn_scores * time_decay.unsqueeze(0).unsqueeze(0)

        # Normalize and compute attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)

        return output

class FourierEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_frequencies=50, device=device):
        """
        Fourier-Based Embedding Layer
        - vocab_size: Number of tokens
        - embedding_dim: Desired embedding size
        - num_frequencies: Number of Fourier components used (must match embedding_dim or be projected)
        - device: Ensures tensors are on the correct device
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies
        self.device = device

        # Learnable Fourier coefficients for sine and cosine
        self.a_n = nn.Parameter(torch.randn(vocab_size, num_frequencies, device=device))
        self.b_n = nn.Parameter(torch.randn(vocab_size, num_frequencies, device=device))

        # Frequency scaling factors (move to device)
        self.freqs = torch.linspace(1, num_frequencies, num_frequencies, device=device).view(1, -1)

        # 🔹 Projection layer to ensure output matches `embedding_dim`
        self.projection = nn.Linear(num_frequencies, embedding_dim)

    def forward(self, token_ids):
        """
        Generate embeddings dynamically using Fourier Series
        - token_ids: Tensor of token indices (batch, seq_len)
        """
        batch_size, seq_len = token_ids.shape

        # Normalize token IDs to continuous space
        x = token_ids.float().unsqueeze(-1) / self.vocab_size  # Shape: (batch, seq_len, 1)

        # Ensure `self.freqs` is on the same device as token_ids
        self.freqs = self.freqs.to(token_ids.device)

        # Compute Fourier embedding
        cos_terms = torch.cos(2 * math.pi * self.freqs * x)  # (batch, seq_len, num_frequencies)
        sin_terms = torch.sin(2 * math.pi * self.freqs * x)  # (batch, seq_len, num_frequencies)

        # Multiply by learnable coefficients
        embedding = (self.a_n[token_ids] * cos_terms + self.b_n[token_ids] * sin_terms)  # (batch, seq_len, num_frequencies)

        # 🔹 Ensure output size matches `embedding_dim` by projecting
        embedding = self.projection(embedding)  # (batch, seq_len, embedding_dim)

        return embedding


class HierarchicalEmbedding(nn.Module):
    def __init__(self, base_vocab_size, embedding_dim, max_levels=3, max_length=30):
        super().__init__()
        self.base_vocab_size = base_vocab_size
        self.embedding_dim = embedding_dim
        self.max_levels = max_levels

        # 🔹 Hierarchical embeddings at different abstraction levels
        self.embeddings = nn.ModuleList([
            nn.Embedding(base_vocab_size, embedding_dim) for _ in range(max_levels)
        ])
        self.fourier_pos = FourierPositionalEncoding(embedding_dim, max_length)

    def forward(self, token_ids, level=0):
        """Retrieves embeddings at the specified hierarchical level."""
        if level >= self.max_levels:
            raise ValueError(f"Level {level} exceeds max_levels {self.max_levels}")
        valid_vocab_size = self.embeddings[level].num_embeddings  # Get vocab size for this level
        token_ids = token_ids.clamp(0, valid_vocab_size - 1)  # Ensure all indices are within range
        return self.embeddings[level](token_ids)


class FourierPositionalEncoding(nn.Module):
    """
    Fourier-based positional encoding that combines frequency information
    with standard embeddings to capture richer positional relations.
    """
    def __init__(self, embedding_dim, max_length=30):
        super().__init__()

        # Create a fixed positional encoding matrix
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))

        pos_enc = torch.zeros(max_length, embedding_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        """Apply Fourier positional encoding to the input tensor."""
        return x + self.pos_enc[:x.shape[1], :].unsqueeze(0)
import torch
import torch.nn as nn
import math

class FourierSummaryEmbedding(nn.Module):
    """
    Generates summary embeddings for hierarchical levels using:
    - Learned embeddings for high-level abstraction.
    - Fourier-based encoding for positional awareness.
    """

    def __init__(self, embedding_dim, max_levels, max_length=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_levels = max_levels

        # 🔹 Fourier Positional Encoding
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))

        pos_enc = torch.zeros(max_length, embedding_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_enc", pos_enc)

        # 🔹 Learnable Summary Embeddings
        self.summary_embeddings = nn.Embedding(max_levels, embedding_dim)

    def forward(self, x, level):
        """
        Apply Fourier encoding + learned summary embedding based on level.
        - x: input sequence embeddings
        - level: hierarchical level (0 = lowest, max_levels = highest)
        """
        pos_encoded = x + self.pos_enc[: x.shape[1], :].unsqueeze(0)
        print(pos_encoded.shape)
        level = torch.tensor(level).to(device)
        print(self.summary_embeddings(level).shape)
        level_embedding = self.summary_embeddings(level)#.unsqueeze(1)
        print(level_embedding.shape)
        return pos_encoded + level_embedding  # 🔹 Mix Fourier and learned embeddings

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, hierarchy_levels=3):
        super(TransformerModel, self).__init__()
        self.embed_size = embedding_dim
        self.hierarchy_levels = hierarchy_levels
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 🔹 Use Hierarchical Embedding instead of flat embeddings
        self.hierarchical_embedding = HierarchicalEmbedding(vocab_size, embedding_dim, hierarchy_levels)

        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.encoder_layers = nn.ModuleList([
            HierarchicalMultiHeadLatentAttention(embedding_dim, num_heads, embedding_dim // 2) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, tgt, level=0):
        src_emb = self.embedding(src, level)
        tgt_emb = self.embedding(tgt, level)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
    
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)

        output = self.fc_out(src_emb)
        return output

class Transformer_Model(nn.Module):
    """
    Transformer model that processes input hierarchically,
    summarizing lower levels into Fourier-based embeddings.
    """

    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, hierarchy_levels=3, chunk_size=50):
        super().__init__()
        self.embed_size = embedding_dim
        self.hierarchy_levels = hierarchy_levels
        self.chunk_size = chunk_size

        # 🔹 Token Embedding (Standard Transformer Style)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 🔹 Fourier Summary Embeddings for hierarchical levels
        self.hierarchical_embedding = FourierSummaryEmbedding(embedding_dim, hierarchy_levels)

        # 🔹 Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads) for _ in range(num_layers)
        ])

        # 🔹 Output Projection
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, level=0):
        """
        Processes input hierarchically:
        - src: input token sequences.
        - level: hierarchy level.
        """
        batch_size, seq_len = src.shape

        # 🔹 Token Embeddings
        token_embeddings = self.embedding(src)

        # 🔹 Chunking input into hierarchical units
        num_chunks = (seq_len // self.chunk_size)
        hierarchical_chunks = token_embeddings.view(batch_size, num_chunks, self.chunk_size, self.embed_size)

        # 🔹 Generate summary embeddings per chunk
        summaries = torch.mean(hierarchical_chunks, dim=2)  # 🔹 Average embeddings per chunk
        hierarchical_embeddings = self.hierarchical_embedding(summaries, level)  # 🔹 Apply Fourier encoding

        # 🔹 Pass through transformer layers
        for layer in self.encoder_layers:
            hierarchical_embeddings = layer(hierarchical_embeddings)

        # 🔹 Final projection
        output = self.fc_out(hierarchical_embeddings.view(batch_size, -1, self.embed_size))

        return output


########################################
# 5. Training Loop
########################################


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for batch_idx, (src, target) in enumerate(dataloader):
        
        src = src.to(device)
        target = target.to(device)

        #decoder_input = target[:, :-1]  # Remove last token from target to match output shape
        target_labels = target[:, 1:]  # Shift target labels by one position

        optimizer.zero_grad()
        
        # 🔹 Get predictions & rule-modified embeddings
        #output = model(src, decoder_input)
        output = model(src)
        # 🔹 Ensure `output` and `target_labels` have the same sequence length
        seq_len = min(output.shape[1], target_labels.shape[1])  # Get the shorter sequence length
        output = output[:, :seq_len, :]  # Truncate logits if too long
        target_labels = target_labels[:, :seq_len]  # Truncate targets if too long

        # 🔹 Flatten for cross_entropy()
        loss = criterion(output.reshape(-1, output.shape[-1]), target_labels.reshape(-1))
        n+=1
        print(f"Iteration {n}, Loss: {loss.item()}")
        if torch.isnan(loss) or torch.isinf(loss):
            print("🚨 Warning: NaN or Inf detected in loss! Skipping update.")
            return

        loss.backward()

        # 🔹 Track how rules affected loss
        prev_loss = loss.item()
        # Clip gradients to prevent exploding values
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 🔹 After updating, re-run forward to see new loss
        with torch.no_grad():
            #output_new = model(src, decoder_input)
            output_new = model(src)
            new_loss = criterion(output_new[:, :seq_len, :].reshape(-1, output_new.shape[-1]), 
                                 target_labels.reshape(-1)).item()
            loss_diff = prev_loss - new_loss  # Negative means rule improved loss

            # 🔹 Update rule effectiveness
            model.embedding.update_rule_scores(src, loss_diff)

        # 🔹 Occasionally add a new rule
        if batch_idx % 50 == 0:  # Every 50 batches, check for new rules
            model.embedding.add_new_rule()

        total_loss += loss.item()
    
    return total_loss / len(dataloader)


########################################
#6. inference
########################################

def hierarchical_inference(model, input_text, max_seq_length, device):
    """
    Processes input hierarchically and generates responses in a structured manner.
    """
    model.eval()
    hierarchical_tokenizer = HierarchicalTokenizer(tokenizer)

    # 🔹 Tokenize input into hierarchical chunks
    hierarchical_tokens = hierarchical_tokenizer.tokenize(input_text)

    generated_output = []
    for level in range(3):  # Iterate over hierarchy levels
        with torch.no_grad():
            input_ids = torch.tensor(hierarchical_tokens, dtype=torch.long).unsqueeze(0).to(device)
            output = model(input_ids, input_ids, level=level)  # 🔹 Pass hierarchy level

            # Decode the output
            generated_tokens = torch.argmax(output, dim=-1).squeeze(0).tolist()
            generated_output.append(hierarchical_tokenizer.decode(generated_tokens))

    return " ".join(generated_output)

# Inference function for autoregressive decoding.
def inference(model, input_text, max_seq_length, device, max_generated=30):
                    model.eval()
                    with torch.no_grad():
                        # Tokenize the prompt and move to the correct device.
                        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
                        # Pad input_ids to the maximum sequence length
                        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                        generated_text = input_ids
                        generated = []
                        logging.debug(f"Padded input_ids Shape: {input_ids.shape}")

                        # Choose a start token for the dummy target.
                        # Here we use tokenizer.eos_token_id if available; otherwise, fallback to tokenizer.pad_token_id.
                        bos_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
                        eos_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                        eos_token  = torch.tensor([[eos_token]], device=device)

                        tgt_ids = torch.tensor([[bos_token]], device=device)
                        tgt_ids = torch.cat([tgt_ids, input_ids], dim=1)
                        logging.info(f"tgt_ids: {tgt_ids}")

                        # Keep track of the original input length
                        input_length = input_ids.size(1)

                        for _ in range(seq_len - input_ids.size(1)):
                            # Generate the target mask for the current target sequence length.
                            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
                            # Forward pass through the model
                            outputs = model(input_ids, tgt_ids)
                            logging.debug(f"output shape: {outputs.shape}")

                            # Get logits for the last token and apply argmax to get the next token ID
                            next_token_logits = outputs[:, -1, :]  # Get the logits for the last position
                            repetition_penalty = 1.2  # Adjust for stronger penalty
                            # Apply repetition penalty while excluding special tokens like PAD (0)
                            for token in set(generated_text[0].tolist()):
                                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                                    next_token_logits[0, token] /= repetition_penalty


                            top_p = 0.9  # Cumulative probability threshold
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0

                            filtered_logits = next_token_logits.clone()
                            filtered_logits[sorted_indices_to_remove] = float('-inf')

                            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                            logging.debug(f"next_token_logits: {next_token_logits}")
                            logging.debug(f"next_token_logits shape: {next_token_logits.shape}")
                            logging.debug(f"next_token_id shape: {next_token_id.shape}")
                            logging.debug(f"next_token_id: {next_token_id}")
                            # Append the new token to the target sequence.
                            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
                            logging.debug(f"tgt_ids: {tgt_ids}")
                            input_ids = input_ids[input_ids != tokenizer.pad_token_id].unsqueeze(0)
                            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                            logging.debug(f"input_ids: {input_ids}")
                            generated.append(tokenizer.decode(next_token_id[0].tolist()))
                            logging.debug(f"generated_text: {generated_text}")
                            #print(tgt_ids)
                            # Stop generation if eos_token is generated
                            if next_token_id.item() == eos_token or tgt_ids.size(1) >= max_seq_length:
                                break

                    return generated


def load_json_file(file_path):
    """Load the JSON dataset file properly."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # 🔹 Ensure it's properly parsed
            if not isinstance(data, list):
                raise ValueError("🚨 Loaded data is not a list of dictionaries.")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"🚨 Failed to parse JSON: {e}")


########################################
# 7. Main Function
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r"C:\Users\Austin\.cursor\ruletransformer-main\mhlatest\skyt1sample.json", help='Path to JSON data')
    parser.add_argument('--epochs', type=int, default=105, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--max_seq_length', type=int, default=500, help='Fixed maximum sequence length')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # ***** NEW: Load tokenizer from file instead of building from the data *****

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    # Load dataset correctly
    json_data = load_json_file(args.data)

    # Pass parsed JSON instead of raw file path
    dataset = ChatDataset(json_data, tokenizer, args.max_seq_length)

    # 🔹 Ensure dataset isn't empty
    if len(dataset) == 0:
        raise ValueError("🚨 Dataset is empty after filtering invalid entries! Check your dataset.")

    # Use a lambda to pass the fixed length to collate_fn.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, args.max_seq_length, tokenizer))
    
    embed_size = 100
    num_heads = 10
    num_layers = 2
    seq_length = args.max_seq_length
    # Initialize the integrated model with desired module toggles.
    model = Transformer_Model(vocab_size, embed_size, num_layers, num_heads, seq_length=args.max_seq_length).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    for epoch in range(1, args.epochs + 1):
        #avg_loss = train_model(model, dataloader, optimizer, criterion, device)
        avg_loss = train_model(model, dataloader, optimizer, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Set the model to evaluation mode and perform inference.
    prompt = "What is the critical temperature of a superconducting thin film made of lead with a thickness of 100 nm?"
    generated_text = hierarchical_inference(model, prompt, seq_length, device)
    print("Generated text:")
    print(generated_text)

if __name__ == '__main__':
    main()
