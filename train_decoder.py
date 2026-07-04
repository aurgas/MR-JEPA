"""
train_decoder.py - Phase 2 Training for JEPA Decoder
Freezes the trained JEPA Encoders & Predictor and trains the Autoregressive Decoder.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from pathlib import Path

# Import Phase 1 Models and Data pipeline
from train import GSM8KDataset, JEPA_ReasoningModel
from decoder import JEPADecoder


class JEPAPipeline(nn.Module):
    """
    Combines the trained JEPA model (frozen) with the trainable Decoder for Phase 2.
    """
    def __init__(self, jepa_model, decoder):
        super().__init__()
        self.jepa = jepa_model
        
        # Freeze all Phase 1 JEPA parameters
        for param in self.jepa.parameters():
            param.requires_grad = False
        self.jepa.eval()
        
        # Trainable Decoder
        self.decoder = decoder
        
    def forward(self, batch):
        """
        Forward pass for Teacher Forced training.
        """
        # 1. Get predicted_emb from frozen JEPA (no gradients needed for Phase 1 backprop)
        with torch.no_grad():
            predicted_emb, _ = self.jepa(batch)
            
        # 2. Forward pass through decoder
        # For teacher forcing, we pass the ground truth answer sequence
        logits = self.decoder(
            input_ids=batch['answer_ids'], 
            predicted_emb=predicted_emb,
            attention_mask=batch['answer_mask']
        )
        return logits


def train_decoder():
    # Phase 2 Configuration
    config = {
        'vocab_size': 50257,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 3,
        'd_ff': 2048,
        'dropout': 0.1,
        'predictor_heads': 4,
        'predictor_d_ff': 1024,
        'max_seq_len_question': 128,
        'max_seq_len_cot': 320,
        'max_seq_len_answer': 32,
        'y_encoder_use_context': True,
        'loss_type': 'hybrid',
        
        # Decoder specific
        'decoder_layers': 2,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'epochs': 5,
        'val_split': 0.1,
        
        # Paths
        'train_csv': 'gsm8k_train.csv',
        'jepa_checkpoint': './checkpoints/checkpoint_epoch_9.pt',
        'save_dir': './checkpoints_decoder',
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    config['vocab_size'] = len(tokenizer)
    
    # 2. Prepare Data
    print("\nLoading and splitting data...")
    try:
        full_train_dataset = GSM8KDataset(config['train_csv'], tokenizer, config, split='full_train')
    except Exception as e:
        print(f"Dataset load failed: {e}. Generating dummy dataset for structural validation.")
        return
        
    train_size = int((1 - config['val_split']) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducibility
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 3. Load Phase 1 Model
    print("\nLoading JEPA Phase 1 Checkpoint...")
    jepa_model = JEPA_ReasoningModel(config).to(device)
    if os.path.exists(config['jepa_checkpoint']):
        checkpoint = torch.load(config['jepa_checkpoint'], map_location=device, weights_only=False)
        jepa_model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ JEPA Checkpoint successfully loaded.")
    else:
        print(f"⚠️ Warning: JEPA Checkpoint {config['jepa_checkpoint']} not found! Using random initialization for testing.")
        
    # 4. Initialize Phase 2 Pipeline
    decoder = JEPADecoder(config).to(device)
    pipeline = JEPAPipeline(jepa_model, decoder).to(device)
    
    optimizer = AdamW(pipeline.decoder.parameters(), lr=config['learning_rate'])
    # GPT2 pad_token == eos_token. Do not ignore it, so the model learns to output EOS to stop correctly.
    criterion = nn.CrossEntropyLoss()
    
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # 5. Training Loop
    print("\n" + "="*50)
    print("Starting Phase 2 Training (Decoder Teacher Forcing)")
    print("="*50)
    
    for epoch in range(config['epochs']):
        pipeline.decoder.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        
        for batch in pbar:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Autoregressive shifting
            # Input to decoder: tokens 0 to sequence length N-1
            # Targets to predict: tokens 1 to sequence length N
            bos_tokens = torch.full((batch['answer_ids'].size(0), 1), tokenizer.eos_token_id, dtype=torch.long, device=device)
            input_ids = torch.cat([bos_tokens, batch['answer_ids'][:, :-1]], dim=1)
            target_ids = batch['answer_ids']
            
            bos_mask = torch.ones((batch['answer_mask'].size(0), 1), dtype=torch.long, device=device)
            attention_mask = torch.cat([bos_mask, batch['answer_mask'][:, :-1]], dim=1)
            
            optimizer.zero_grad()
            
            # Setup batch for model forward specifically with shifted answering
            batch_for_forward = batch.copy()
            batch_for_forward['answer_ids'] = input_ids
            batch_for_forward['answer_mask'] = attention_mask
            
            logits = pipeline(batch_for_forward)
            
            # Flatten predictions and targets to calculate sequence-wide token loss
            loss = criterion(logits.reshape(-1, config['vocab_size']), target_ids.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 6. Validation Loop
        pipeline.decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                bos_tokens = torch.full((batch['answer_ids'].size(0), 1), tokenizer.eos_token_id, dtype=torch.long, device=device)
                input_ids = torch.cat([bos_tokens, batch['answer_ids'][:, :-1]], dim=1)
                target_ids = batch['answer_ids']
                
                bos_mask = torch.ones((batch['answer_mask'].size(0), 1), dtype=torch.long, device=device)
                attention_mask = torch.cat([bos_mask, batch['answer_mask'][:, :-1]], dim=1)
                
                batch_for_forward = batch.copy()
                batch_for_forward['answer_ids'] = input_ids
                batch_for_forward['answer_mask'] = attention_mask
                
                logits = pipeline(batch_for_forward)
                loss = criterion(logits.reshape(-1, config['vocab_size']), target_ids.reshape(-1))
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} Summary - Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config['save_dir'], 'best_decoder.pt')
            
            # Save the raw decoder state dict (not the whole pipeline) so it easily loads standalone
            torch.save({
                'decoder_state_dict': pipeline.decoder.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, save_path)
            print(f"🌟 New best decoder checkpoint saved to {save_path}\n")

if __name__ == "__main__":
    train_decoder()
