# Test script
import torch
from transformers import AutoTokenizer
from Encoders import *

OPTIMIZED_SMALL_CONFIG = {
    # Vocabulary
    'vocab_size': 50257,  # GPT-2 tokenizer
    
    # Model dimensions
    'd_model': 512,
    'n_heads': 8,          
    'n_layers': 3,         # ← Reduced from 4
    'd_ff': 2048,          # 4 × d_model
    'dropout': 0.1,
    
    # Sequence lengths (different for each input)
    'max_seq_len_question': 128,
    'max_seq_len_cot': 320,      # ← Based on your analysis
    'max_seq_len_answer': 32,
    
    # Y-encoder specific
    'use_simple_y_encoder': True,  # ← Use positional encoding only
}

# Config
config = OPTIMIZED_SMALL_CONFIG

# Initialize
tokenizer = AutoTokenizer.from_pretrained('gpt2')
config['vocab_size'] = len(tokenizer)
tokenizer.pad_token = tokenizer.eos_token

encoders = JEPA_Encoders(config)

# Sample data
question = "Janet's ducks lay 16 eggs per day. She eats 3 and uses 4. How much does she make selling the rest at $2 each?"
cot = "Janet sells 16 - 3 - 4 = 9 duck eggs. She makes 9 * 2 = 18 dollars."
answer = "18"

# Tokenize
q_tok = tokenizer(question, return_tensors='pt', padding=True, 
                  truncation=True, max_length=128)
cot_tok = tokenizer(cot, return_tensors='pt', padding=True,
                    truncation=True, max_length=320)
a_tok = tokenizer(answer, return_tensors='pt', padding=True,
                  truncation=True, max_length=32)

# Encode
q_emb, cot_emb, a_emb = encoders(
    question_ids=q_tok['input_ids'],
    cot_ids=cot_tok['input_ids'],
    answer_ids=a_tok['input_ids'],
    question_mask=q_tok['attention_mask'],
    cot_mask=cot_tok['attention_mask'],
    answer_mask=a_tok['attention_mask']
)

print(f"Question embedding: {q_emb.shape}")  # [1, 512]
print(f"CoT embedding: {cot_emb.shape}")     # [1, 512]
print(f"Answer embedding: {a_emb.shape}")    # [1, 512]