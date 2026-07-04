"""
encoders.py - JEPA Reasoning Model Encoders with RoPE
Implements Question, CoT, and Y encoders with Rotary Position Embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContextAwareYEncoder(nn.Module):
    """
    Context-aware Y-encoder that conditions answer embeddings on the question
    This ensures same numerical answers have different embeddings based on context
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config['d_model']
        max_seq_len = config['max_seq_len_answer']
        
        # Token embedding for answers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        
        # RoPE for positional encoding
        self.rope = RotaryPositionalEmbedding(
            d_model=config['d_model'],
            max_seq_len=max_seq_len,
            base=config.get('rope_base', 10000)
        )
        
        # Cross-attention: answer attends to question context (optional)
        self.use_context = config.get('y_encoder_use_context', False)
        if self.use_context:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config['d_model'],
                num_heads=4,
                dropout=config['dropout'],
                batch_first=True
            )
            self.norm_attn = nn.LayerNorm(config['d_model'])
        
        # Processing layers
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model'] * 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'] * 2, config['d_model']),
            nn.Dropout(config['dropout'])
        )
        
        self.norm_ffn = nn.LayerNorm(config['d_model'])
        
        # Output projection
        self.output_projection = nn.Linear(config['d_model'], config['d_model'])
        
        self.dropout = nn.Dropout(config['dropout'])
        self.norm_out = nn.LayerNorm(config['d_model'])
    
    def forward(self, input_ids, attention_mask=None, context_emb=None):
        """
        Forward pass with optional context conditioning
        
        Args:
            input_ids: [batch_size, seq_len] - answer tokens
            attention_mask: [batch_size, seq_len]
            context_emb: [batch_size, d_model] - optional question embedding for context
        
        Returns:
            answer_emb: [batch_size, d_model]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [B, L, d_model]
        token_embeds = token_embeds * math.sqrt(self.d_model)
        
        # Apply RoPE
        embeddings = self.rope(token_embeds)  # [B, L, d_model]
        embeddings = self.dropout(embeddings)
        
        # Optional: Cross-attention with context (question)
        if self.use_context and context_emb is not None:
            # Answer attends to question context
            query = embeddings  # [B, L, d_model]
            key_value = context_emb.unsqueeze(1)  # [B, 1, d_model]
            
            attn_out, _ = self.cross_attention(query, key_value, key_value)
            embeddings = self.norm_attn(embeddings + attn_out)
        
        # Mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            answer_emb = sum_embeddings / sum_mask
        else:
            answer_emb = embeddings.mean(dim=1)  # [B, d_model]
        
        # Feed-forward processing
        ffn_out = self.ffn(answer_emb)
        answer_emb = self.norm_ffn(answer_emb + ffn_out)
        
        # Final projection
        answer_emb = self.output_projection(answer_emb)
        answer_emb = self.norm_out(answer_emb)
        
        return answer_emb

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Used in modern transformers like LLaMA, GPT-NeoX
    """
    
    def __init__(self, d_model, max_seq_len=2048, base=10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute frequencies
        self._precompute_freqs_cis(max_seq_len)
    
    def _precompute_freqs_cis(self, seq_len):
        """
        Precompute cos and sin frequencies for the maximum sequence length
        """
        # Position indices
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies: outer product [seq_len, d_model/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Create complex representation e^(i*theta) safely for MPS
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        freqs_cis = torch.complex(freqs_cos, freqs_sin)
        
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
    
    def forward(self, x):
        """
        Apply rotary position embedding
        
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            x_rotated: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Extend precomputed values if needed
        if seq_len > self.freqs_cis.shape[0]:
            self._precompute_freqs_cis(seq_len)
        
        # Reshape to work with complex numbers: [B, L, d_model/2, 2]
        x_reshaped = x.reshape(batch_size, seq_len, -1, 2)
        
        # Convert to complex: [B, L, d_model/2]
        x_complex = torch.view_as_complex(x_reshaped.float())
        
        # Get frequencies for this sequence length: [L, d_model/2]
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Apply rotation (complex multiplication)
        # Broadcast freqs_cis to [1, L, d_model/2]
        x_rotated_complex = x_complex * freqs_cis.unsqueeze(0)
        
        # Convert back to real: [B, L, d_model/2, 2] -> [B, L, d_model]
        x_rotated = torch.view_as_real(x_rotated_complex).reshape(batch_size, seq_len, d_model)
        
        return x_rotated.type_as(x)


class TextEmbeddingWithRoPE(nn.Module):
    """
    Text embedding layer with Rotary Position Embeddings
    Replaces traditional learnable positional embeddings
    """
    
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1, rope_base=10000):
        super().__init__()
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # RoPE (no learnable parameters for positions)
        self.rope = RotaryPositionalEmbedding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            base=rope_base
        )
        
        # [CLS] token for sequence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch_size, seq_len]
        
        Returns:
            embeddings: [batch_size, seq_len+1, d_model] (with [CLS] prepended)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [B, L, d_model]
        
        # Scale embeddings (as per "Attention is All You Need")
        token_embeds = token_embeds * math.sqrt(self.d_model)
        
        # Apply RoPE
        embeddings = self.rope(token_embeds)  # [B, L, d_model]
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # [B, L+1, d_model]
        
        return self.dropout(embeddings)


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feed-forward
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
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
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] - padding mask (True = ignore)
        
        Returns:
            x: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class UnifiedEncoder(nn.Module):
    """
    Unified encoder architecture with RoPE
    Used for both Question and CoT encoders
    """
    
    def __init__(self, config, max_seq_len):
        super().__init__()
        
        self.config = config
        
        # Embedding layer with RoPE
        self.embedding = TextEmbeddingWithRoPE(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            max_seq_len=max_seq_len,
            dropout=config['dropout'],
            rope_base=config.get('rope_base', 10000)
        )
        
        # Stack of transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                dropout=config['dropout']
            )
            for _ in range(config['n_layers'])
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config['d_model'], config['d_model'])
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding
        
        Returns:
            latent_embedding: [batch_size, d_model]
        """
        # Get embeddings with RoPE
        x = self.embedding(input_ids)  # [B, L+1, d_model] (L+1 due to [CLS])
        
        # Create padding mask for attention
        if attention_mask is not None:
            # Prepend 1 for [CLS] token (always valid)
            cls_mask = torch.ones(
                attention_mask.shape[0], 1, 
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
            
            # Convert to mask format (True = ignore, False = attend)
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=padding_mask)
        
        # Extract [CLS] token representation (first token)
        cls_output = x[:, 0, :]  # [B, d_model]
        
        # Project to final embedding space
        latent_embedding = self.output_projection(cls_output)
        
        return latent_embedding


class PositionalYEncoder(nn.Module):
    """
    Lightweight Y-encoder with RoPE
    No transformer layers - just embedding + RoPE + pooling
    Efficient for short answer sequences
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config['d_model']
        max_seq_len = config['max_seq_len_answer']
        
        # Token embedding
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(
            d_model=config['d_model'],
            max_seq_len=max_seq_len,
            base=config.get('rope_base', 10000)
        )
        
        # Output projection
        self.output_projection = nn.Linear(config['d_model'], config['d_model'])
        
        self.dropout = nn.Dropout(config['dropout'])
        self.norm = nn.LayerNorm(config['d_model'])
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            answer_emb: [batch_size, d_model]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [B, L, d_model]
        token_embeds = token_embeds * math.sqrt(self.d_model)
        
        # Apply RoPE
        embeddings = self.rope(token_embeds)  # [B, L, d_model]
        embeddings = self.dropout(embeddings)
        
        # Mean pooling over sequence (answers are short)
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            answer_emb = sum_embeddings / sum_mask
        else:
            answer_emb = embeddings.mean(dim=1)  # [B, d_model]
        
        # Project and normalize
        answer_emb = self.output_projection(answer_emb)
        answer_emb = self.norm(answer_emb)
        
        return answer_emb


class JEPA_Encoders(nn.Module):
    """
    Complete encoder module for JEPA reasoning model
    Contains:
    - Question Encoder (full transformer with RoPE)
    - CoT Encoder (full transformer with RoPE)
    - Y-Encoder (context-aware with RoPE)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Question encoder
        self.question_encoder = UnifiedEncoder(
            config,
            max_seq_len=config['max_seq_len_question']
        )
        
        # CoT encoder
        self.cot_encoder = UnifiedEncoder(
            config,
            max_seq_len=config['max_seq_len_cot']
        )
        
        # Y encoder (context-aware)
        self.y_encoder = ContextAwareYEncoder(config)
        
        self._print_model_info()
    
    def _print_model_info(self):
        """Print parameter counts for each encoder"""
        q_params = sum(p.numel() for p in self.question_encoder.parameters()) / 1e6
        cot_params = sum(p.numel() for p in self.cot_encoder.parameters()) / 1e6
        y_params = sum(p.numel() for p in self.y_encoder.parameters()) / 1e6
        total = q_params + cot_params + y_params
        
        print(f"\n{'='*70}")
        print(f"JEPA Encoder Architecture (with RoPE)")
        print(f"{'='*70}")
        print(f"Question Encoder:  {q_params:>6.1f}M params (seq_len={self.config['max_seq_len_question']})")
        print(f"CoT Encoder:       {cot_params:>6.1f}M params (seq_len={self.config['max_seq_len_cot']})")
        print(f"Y Encoder:         {y_params:>6.1f}M params (seq_len={self.config['max_seq_len_answer']}, context-aware)")
        print(f"{'-'*70}")
        print(f"Total Parameters:  {total:>6.1f}M")
        print(f"Position Encoding: RoPE (Rotary Position Embedding)")
        print(f"RoPE Base:         {self.config.get('rope_base', 10000)}")
        print(f"Y-Encoder Context: {'Enabled' if self.config.get('y_encoder_use_context', False) else 'Disabled'}")
        print(f"{'='*70}\n")
    
    def forward(self, question_ids, cot_ids, answer_ids=None,
                question_mask=None, cot_mask=None, answer_mask=None):
        """
        Encode all inputs
        
        Args:
            question_ids: [batch_size, L_q]
            cot_ids: [batch_size, L_c]
            answer_ids: [batch_size, L_a] (optional, for training)
            question_mask: [batch_size, L_q]
            cot_mask: [batch_size, L_c]
            answer_mask: [batch_size, L_a]
        
        Returns:
            question_emb: [batch_size, d_model]
            cot_emb: [batch_size, d_model]
            answer_emb: [batch_size, d_model] (if answer_ids provided)
        """
        # Encode question
        question_emb = self.question_encoder(question_ids, question_mask)
        
        # Encode CoT
        cot_emb = self.cot_encoder(cot_ids, cot_mask)
        
        # Encode answer (only during training)
        answer_emb = None
        if answer_ids is not None:
            # We wrap the Y-encoder in torch.no_grad() to natively apply the stop-gradient.
            # This completely bypasses building a massive 28M parameter computation graph 
            # that gets thrown away, greatly speeding up training and preventing GC thrashing.
            with torch.no_grad():
                # Pass question context to Y-encoder if enabled
                if self.config.get('y_encoder_use_context', False):
                    answer_emb = self.y_encoder(answer_ids, answer_mask, context_emb=question_emb.detach())
                else:
                    answer_emb = self.y_encoder(answer_ids, answer_mask)
        
        return question_emb, cot_emb, answer_emb

def test_encoders():
    """
    Test script to verify encoders work correctly
    """
    print("Testing JEPA Encoders with RoPE...")
    
    # Configuration
    config = {
        'vocab_size': 50257,         # GPT-2 vocab size
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 3,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_seq_len_question': 128,
        'max_seq_len_cot': 320,
        'max_seq_len_answer': 32,
        'rope_base': 10000,
    }
    
    # Initialize encoders
    encoders = JEPA_Encoders(config)
    encoders.eval()
    
    # Create dummy inputs
    batch_size = 4
    question_len = 50
    cot_len = 100
    answer_len = 5
    
    question_ids = torch.randint(0, config['vocab_size'], (batch_size, question_len))
    cot_ids = torch.randint(0, config['vocab_size'], (batch_size, cot_len))
    answer_ids = torch.randint(0, config['vocab_size'], (batch_size, answer_len))
    
    # Create attention masks (no padding for this test)
    question_mask = torch.ones(batch_size, question_len)
    cot_mask = torch.ones(batch_size, cot_len)
    answer_mask = torch.ones(batch_size, answer_len)
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        question_emb, cot_emb, answer_emb = encoders(
            question_ids=question_ids,
            cot_ids=cot_ids,
            answer_ids=answer_ids,
            question_mask=question_mask,
            cot_mask=cot_mask,
            answer_mask=answer_mask
        )
    
    # Print results
    print(f"\n{'='*70}")
    print("Output Shapes:")
    print(f"{'='*70}")
    print(f"Question embedding: {question_emb.shape}")  # [4, 512]
    print(f"CoT embedding:      {cot_emb.shape}")       # [4, 512]
    print(f"Answer embedding:   {answer_emb.shape}")    # [4, 512]
    print(f"{'='*70}")
    
    # Test RoPE functionality
    print("\nTesting RoPE functionality...")
    
    # Same token at different positions should have different embeddings
    same_token_ids = torch.full((1, 10), fill_value=42)
    same_token_mask = torch.ones(1, 10)
    
    with torch.no_grad():
        emb, _, _ = encoders(same_token_ids, same_token_ids, None,
                            same_token_mask, same_token_mask, None)
    
    # Get embeddings from the encoder before [CLS] extraction
    with torch.no_grad():
        embeddings_with_rope = encoders.question_encoder.embedding(same_token_ids)
    
    # Check difference between positions
    pos1 = embeddings_with_rope[0, 1, :]  # Position 1 (after [CLS])
    pos5 = embeddings_with_rope[0, 5, :]  # Position 5
    difference = (pos1 - pos5).abs().mean()
    
    print(f"Average difference between same token at different positions: {difference:.4f}")
    if difference > 0.01:
        print("✓ RoPE is working correctly!")
    else:
        print("⚠ Warning: RoPE might not be applied properly")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_encoders()