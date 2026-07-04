"""
decoder.py - Autoregressive Transformer Decoder for R-JEPA
Maps the continuous predicted latent embedding back to discrete text answers.
"""

import torch
import torch.nn as nn
import math

# Reusing the RoPE from Encoders
from Encoders import RotaryPositionalEmbedding

class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder layer with Causal Self-Attention and Cross-Attention
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Causal Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-Attention (attends to JEPA predicted_emb)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask=None, padding_mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model] - decoder input (previously generated tokens)
            memory: [batch_size, 1, d_model] - JEPA predicted embedding
            causal_mask: [seq_len, seq_len] - mask to prevent looking ahead
            padding_mask: [batch_size, seq_len] - True for pad tokens
        """
        # 1. Causal Self-Attention
        attn_output, _ = self.self_attn(
            x, x, x, 
            attn_mask=causal_mask,
            key_padding_mask=padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Cross-Attention (Decoder attends to JEPA Memory)
        # memory is the predicted_emb [B, 1, d_model]
        attn_output, _ = self.cross_attn(
            query=x,
            key=memory,
            value=memory
        )
        x = self.norm2(x + self.dropout(attn_output))
        
        # 3. Feed-Forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x

class JEPADecoder(nn.Module):
    """
    Autoregressive Decoder converting JEPA latent embeddings to text tokens.
    Uses RoPE for consistency with the Encoders.
    """
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        max_seq_len = config['max_seq_len_answer']
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional encoding (RoPE)
        self.rope = RotaryPositionalEmbedding(
            d_model=self.d_model,
            max_seq_len=max_seq_len,
            base=config.get('rope_base', 10000)
        )
        
        self.dropout = nn.Dropout(config['dropout'])
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=self.d_model,
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                dropout=config['dropout']
            )
            for _ in range(config.get('decoder_layers', 2)) # Defaulting to lighter decoder
        ])
        
        # Final output projection to vocabulary
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Tie weights between embedding and projection (standard NLP practice)
        self.output_projection.weight = self.token_embedding.weight
        
        self._print_model_info()
        
    def _print_model_info(self):
        params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"\n{'='*70}")
        print(f"JEPA Autoregressive Decoder")
        print(f"{'='*70}")
        print(f"Vocab size:       {self.vocab_size}")
        print(f"Dimension:        {self.d_model}")
        print(f"Layers:           {len(self.layers)}")
        print(f"Total Parameters: {params:.1f}M")
        print(f"{'='*70}\n")
        
    def generate_causal_mask(self, seq_len, device):
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        Used to prevent self-attention from looking into the future.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, input_ids, predicted_emb, attention_mask=None):
        """
        Forward pass for training (Teacher Forcing).
        
        Args:
            input_ids: [batch_size, seq_len] - Target token IDs (shifted right in training loop)
            predicted_emb: [batch_size, d_model] - Output from JEPA predictor
            attention_mask: [batch_size, seq_len] - Mask for padding (False for valid, True for pad)
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Embed tokens
        x = self.token_embedding(input_ids) # [B, L, d_model]
        x = x * math.sqrt(self.d_model)
        
        # 2. Apply RoPE
        x = self.rope(x)
        x = self.dropout(x)
        
        # 3. Create masks
        causal_mask = self.generate_causal_mask(seq_len, device)
        padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # Format JEPA embedding for cross-attention
        # Cross-attention expects memory to have shape [batch, memory_seq_len, d_model]
        memory = predicted_emb.unsqueeze(1) # [B, 1, d_model]
        
        # 4. Pass through decoder layers
        for layer in self.layers:
            x = layer(
                x, 
                memory=memory, 
                causal_mask=causal_mask,
                padding_mask=padding_mask
            )
            
        # 5. Project to vocabulary space
        logits = self.output_projection(x) # [B, seq_len, vocab_size]
        
        return logits
        
    def generate(self, predicted_emb, bos_token_id, eos_token_id, max_new_tokens=32):
        """
        Autoregressive generation at inference time.
        
        Args:
            predicted_emb: [batch_size, d_model] - Embedding to decode from
            bos_token_id: int - Beginning of sequence token ID
            eos_token_id: int - End of sequence token ID
            max_new_tokens: int - Maximum length of generation
            
        Returns:
            generated_ids: [batch_size, max_new_tokens]
        """
        batch_size = predicted_emb.shape[0]
        device = predicted_emb.device
        
        self.eval()
        
        # Start with just the BOS token
        current_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # Format memory
        memory = predicted_emb.unsqueeze(1) # [B, 1, d_model]
        
        # Keep track of batches that have finished generating (hit EOS)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                seq_len = current_ids.shape[1]
                
                # Embed current sequence
                x = self.token_embedding(current_ids)
                x = x * math.sqrt(self.d_model)
                x = self.rope(x)
                
                causal_mask = self.generate_causal_mask(seq_len, device)
                
                # Pass through layers
                for layer in self.layers:
                    x = layer(
                        x, 
                        memory=memory, 
                        causal_mask=causal_mask,
                        padding_mask=None
                    )
                
                # We only care about the prediction for the LAST token in the sequence
                next_token_logits = self.output_projection(x[:, -1, :])
                
                # Greedy decoding (argmax)
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Update unfinished sequences (if next_token is EOS, mark as finished)
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
                
                # Prevent adding tokens to finished sequences (replace with padding)
                next_tokens = next_tokens * unfinished_sequences + eos_token_id * (1 - unfinished_sequences)
                
                # Append next token to current sequence
                current_ids = torch.cat([current_ids, next_tokens.unsqueeze(-1)], dim=-1)
                
                # Stop if all batches have generated EOS
                if unfinished_sequences.max() == 0:
                    break
                    
        return current_ids

def test_decoder():
    """
    Test script to verify decoder dimension correctness and generation.
    """
    print("Testing JEPA Autoregressive Decoder...")
    
    config = {
        'vocab_size': 50000,
        'd_model': 512,
        'n_heads': 8,
        'decoder_layers': 2,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_seq_len_answer': 32,
        'rope_base': 10000,
    }
    
    decoder = JEPADecoder(config)
    batch_size = 4
    seq_len = 10
    
    # Dummy inputs
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    predicted_emb = torch.randn(batch_size, config['d_model'])
    
    # 1. Test Training Forward Pass
    print("\n--- Testing Training Forward Pass ---")
    logits = decoder(input_ids, predicted_emb, attention_mask)
    print(f"Logits shape: {logits.shape} (Expected: [{batch_size}, {seq_len}, {config['vocab_size']}])")
    
    # 2. Test Generation
    print("\n--- Testing Autoregressive Generation ---")
    bos_token_id = 0
    eos_token_id = 1
    generated_ids = decoder.generate(predicted_emb, bos_token_id, eos_token_id, max_new_tokens=5)
    print(f"Generated shape: {generated_ids.shape}")
    print(f"Generated IDs sample:\\n{generated_ids[0]}")
    
    print("\n✓ Decoder tests passed!")

if __name__ == "__main__":
    test_decoder()
