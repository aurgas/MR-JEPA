"""
train.py - Complete Training Pipeline for JEPA Reasoning Model
Uses proper train/validation split with context-aware Y-encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os
from pathlib import Path
import time

# Import our modules
from Encoders import JEPA_Encoders
from predictor import PredictorWithLoss, EvaluationMetrics


class GSM8KDataset(Dataset):
    """
    Dataset class for GSM8K mathematical reasoning
    """
    
    def __init__(self, csv_path, tokenizer, config, split='train'):
        """
        Args:
            csv_path: Path to CSV file (train or test)
            tokenizer: HuggingFace tokenizer
            config: Model configuration
            split: 'train', 'val', or 'test'
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        print(f"\nLoaded {split} dataset: {len(self.df)} examples")
        
        # Show sample
        if len(self.df) > 0:
            print(f"\nSample {split} example:")
            print(f"  Question: {self.df.iloc[0]['question'][:80]}...")
            print(f"  CoT: {self.df.iloc[0]['cot'][:80]}...")
            print(f"  Answer: {self.df.iloc[0]['answer']}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        question = str(row['question'])
        cot = str(row['cot'])
        answer = str(row['answer'])
        
        # Tokenize question
        question_encoded = self.tokenizer(
            question,
            max_length=self.config['max_seq_len_question'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize CoT
        cot_encoded = self.tokenizer(
            cot,
            max_length=self.config['max_seq_len_cot'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize answer
        answer_encoded = self.tokenizer(
            answer,
            max_length=self.config['max_seq_len_answer'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'question_ids': question_encoded['input_ids'].squeeze(0),
            'question_mask': question_encoded['attention_mask'].squeeze(0),
            'cot_ids': cot_encoded['input_ids'].squeeze(0),
            'cot_mask': cot_encoded['attention_mask'].squeeze(0),
            'answer_ids': answer_encoded['input_ids'].squeeze(0),
            'answer_mask': answer_encoded['attention_mask'].squeeze(0),
            'question_text': question,
            'cot_text': cot,
            'answer_text': answer
        }


class JEPA_ReasoningModel(nn.Module):
    """
    Complete JEPA Reasoning Model
    Encoders (8 heads) + Predictor + Loss
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Encoders (Question, CoT, Y)
        self.encoders = JEPA_Encoders(config)
        
        # Predictor with loss
        self.predictor_with_loss = PredictorWithLoss(
            config,
            loss_type=config.get('loss_type', 'cosine')
        )
        
        self._print_total_params()
    
    def _print_total_params(self):
        """Print total model parameters"""
        encoder_params = sum(p.numel() for p in self.encoders.parameters()) / 1e6
        predictor_params = sum(p.numel() for p in self.predictor_with_loss.parameters()) / 1e6
        total_params = encoder_params + predictor_params
        
        print(f"\n{'='*70}")
        print(f"COMPLETE JEPA REASONING MODEL")
        print(f"{'='*70}")
        print(f"Encoders:              {encoder_params:>6.1f}M params")
        print(f"Predictor:             {predictor_params:>6.1f}M params")
        print(f"{'-'*70}")
        print(f"Total Parameters:      {total_params:>6.1f}M")
        print(f"Attention Heads:       {self.config['n_heads']} (encoders)")
        print(f"Predictor Heads:       {self.config.get('predictor_heads', 4)}")
        print(f"Loss Function:         {self.config.get('loss_type', 'cosine').upper()}")
        print(f"Y-Encoder Context:     {'Enabled' if self.config.get('y_encoder_use_context', False) else 'Disabled'}")
        print(f"{'='*70}\n")
    
    def forward(self, batch, return_embeddings=False):
        """
        Forward pass through entire model
        
        Args:
            batch: Dictionary with tokenized inputs
            return_embeddings: Whether to return intermediate embeddings
        
        Returns:
            predicted_emb: Predicted answer embedding
            target_emb: Target answer embedding
            (optional) embeddings_dict: All intermediate embeddings
        """
        # Encode all inputs
        question_emb, cot_emb, target_emb = self.encoders(
            question_ids=batch['question_ids'],
            cot_ids=batch['cot_ids'],
            answer_ids=batch['answer_ids'],
            question_mask=batch['question_mask'],
            cot_mask=batch['cot_mask'],
            answer_mask=batch['answer_mask']
        )
        
        # Predict answer embedding
        predicted_emb = self.predictor_with_loss(question_emb, cot_emb)
        
        if return_embeddings:
            embeddings_dict = {
                'question_emb': question_emb,
                'cot_emb': cot_emb,
                'predicted_emb': predicted_emb,
                'target_emb': target_emb
            }
            return predicted_emb, target_emb, embeddings_dict
        
        return predicted_emb, target_emb
    
    def compute_loss(self, predicted_emb, target_emb):
        """Compute loss"""
        return self.predictor_with_loss.compute_loss(predicted_emb, target_emb)


class Trainer:
    """
    Training manager for JEPA Reasoning Model
    """
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        if config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'] * len(train_loader),
                eta_min=config['learning_rate'] * 0.1
            )
        elif config['scheduler'] == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config['learning_rate'],
                epochs=config['epochs'],
                steps_per_epoch=len(train_loader)
            )
        else:
            self.scheduler = None
        
        # Metrics tracker
        self.metrics = EvaluationMetrics()
        
        # Best model tracking
        self.best_val_similarity = -1.0
        self.best_epoch = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_cosine_sim': [],
            'val_loss': [],
            'val_cosine_sim': [],
            'val_angular_dist': [],
            'learning_rate': []
        }
        
        print(f"\nTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Optimizer: AdamW (lr={config['learning_rate']}, wd={config['weight_decay']})")
        print(f"  Scheduler: {config['scheduler']}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_cos_sim = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predicted_emb, target_emb = self.model(batch)
            
            # Apply stop-gradient to target embedding to prevent representation collapse
            target_emb = target_emb.detach()
            
            # Compute loss
            loss, loss_dict = self.model.compute_loss(predicted_emb, target_emb)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Metrics
            with torch.no_grad():
                cos_sim = self.metrics.cosine_similarity(predicted_emb, target_emb)
            
            total_loss += loss.item()
            total_cos_sim += cos_sim
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cos_sim': f"{cos_sim:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_cos_sim = total_cos_sim / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return avg_loss, avg_cos_sim, current_lr
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_cos_sim = 0
        total_angular_dist = 0
        num_batches = len(self.val_loader)
        
        all_metrics = {
            'cosine_similarity': [],
            'angular_distance': [],
            'accuracy@0.9': [],
            'accuracy@0.8': [],
            'accuracy@0.7': []
        }
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]  ")
        
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predicted_emb, target_emb = self.model(batch)
                
                # Compute loss
                loss, loss_dict = self.model.compute_loss(predicted_emb, target_emb)
                
                # Compute all metrics
                batch_metrics = self.metrics.compute_all_metrics(
                    predicted_emb, target_emb
                )
                
                total_loss += loss.item()
                total_cos_sim += batch_metrics['cosine_similarity']
                total_angular_dist += batch_metrics['angular_distance']
                
                # Accumulate metrics
                for key in all_metrics.keys():
                    all_metrics[key].append(batch_metrics[key])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'cos_sim': f"{batch_metrics['cosine_similarity']:.4f}"
                })
        
        # Average metrics
        val_metrics = {
            'loss': total_loss / num_batches,
            'cosine_similarity': total_cos_sim / num_batches,
            'angular_distance': total_angular_dist / num_batches,
        }
        
        for key in all_metrics.keys():
            val_metrics[key] = np.mean(all_metrics[key])
        
        return val_metrics
    
    def save_checkpoint(self, epoch, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history,
            'best_val_similarity': self.best_val_similarity
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_similarity = checkpoint['best_val_similarity']
        print(f"✓ Checkpoint loaded: {filepath}")
        return checkpoint['epoch']
    
    def train(self, num_epochs, save_dir='./checkpoints'):
        """
        Complete training loop
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}")
        print(f"Save directory: {save_dir}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_cos_sim, current_lr = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_cosine_sim'].append(train_cos_sim)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_cosine_sim'].append(val_metrics['cosine_similarity'])
            self.history['val_angular_dist'].append(val_metrics['angular_distance'])
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs} Summary (Time: {epoch_time:.1f}s)")
            print(f"{'='*70}")
            print(f"Train Loss:           {train_loss:.4f}")
            print(f"Train Cosine Sim:     {train_cos_sim:.4f}")
            print(f"Val Loss:             {val_metrics['loss']:.4f}")
            print(f"Val Cosine Sim:       {val_metrics['cosine_similarity']:.4f}")
            print(f"Val Angular Dist:     {val_metrics['angular_distance']:.2f}°")
            print(f"Val Accuracy@0.9:     {val_metrics['accuracy@0.9']:.2f}%")
            print(f"Val Accuracy@0.8:     {val_metrics['accuracy@0.8']:.2f}%")
            print(f"Val Accuracy@0.7:     {val_metrics['accuracy@0.7']:.2f}%")
            print(f"Learning Rate:        {current_lr:.2e}")
            print(f"{'='*70}\n")
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            self.save_checkpoint(epoch, checkpoint_path)
            
            # Save best model
            if val_metrics['cosine_similarity'] > self.best_val_similarity:
                self.best_val_similarity = val_metrics['cosine_similarity']
                self.best_epoch = epoch + 1
                
                best_path = os.path.join(save_dir, 'best_model.pt')
                self.save_checkpoint(epoch, best_path)
                
                print(f"🌟 New best model! Cosine Sim: {self.best_val_similarity:.4f}")
                print(f"   Saved to: {best_path}\n")
        
        total_time = time.time() - start_time
        
        # Training complete
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val cosine similarity: {self.best_val_similarity:.4f}")
        print(f"{'='*70}\n")
        
        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved: {history_path}")
        
        return self.history


def main():
    """
    Main training script with proper train/val split
    """
    
    # ==================== CONFIGURATION ====================
    config = {
        # Model architecture
        'vocab_size': 50257,        # Will be updated from tokenizer
        'd_model': 512,
        'n_heads': 8,               # 8 heads for encoders
        'n_layers': 3,
        'd_ff': 2048,
        'dropout': 0.1,
        'rope_base': 10000,
        
        # Predictor
        'predictor_heads': 4,
        'predictor_d_ff': 1024,
        
        # Sequence lengths
        'max_seq_len_question': 128,
        'max_seq_len_cot': 320,
        'max_seq_len_answer': 32,
        
        # Loss
        'loss_type': 'contrastive',      # 'cosine', 'contrastive', or 'hybrid'
        'temperature': 1.0,
        
        # Y-Encoder (IMPORTANT: Context-aware encoding)
        'y_encoder_use_context': True,  # Enable context-aware answer encoding
        
        # Training
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'epochs': 10,
        'scheduler': 'cosine',      # 'cosine', 'onecycle', or None
        
        # Data split (IMPORTANT: validation split from training data)
        'val_split': 0.1,           # 10% of training data for validation
        
        # Data paths
        'train_csv': 'gsm8k_train.csv',
        'test_csv': 'gsm8k_test.csv',  # RESERVED FOR FINAL TESTING ONLY
        
        # Checkpoint
        'save_dir': './checkpoints',
        'resume_from': None,        # Path to checkpoint to resume from
    }
    
    # Device (CUDA -> MPS -> CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"JEPA REASONING MODEL - TRAINING")
    print(f"{'='*70}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}\n")
    
    # ==================== TOKENIZER ====================
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    config['vocab_size'] = len(tokenizer)
    print(f"✓ Tokenizer loaded: vocab_size = {config['vocab_size']}")
    
    # ==================== DATASETS ====================
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print(f"{'='*70}")
    print(f"⚠️  NOTE: Test set ({config['test_csv']}) is RESERVED for final evaluation")
    print(f"   Using {config['val_split']*100:.0f}% of training data for validation")
    
    # Load ONLY training data
    full_train_dataset = GSM8KDataset(
        config['train_csv'],
        tokenizer,
        config,
        split='full_train'
    )
    
    # Split training data into train and validation
    train_size = int((1 - config['val_split']) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"\n✓ Data split complete:")
    print(f"  Training:   {len(train_dataset)} examples ({(1-config['val_split'])*100:.0f}%)")
    print(f"  Validation: {len(val_dataset)} examples ({config['val_split']*100:.0f}%)")
    print(f"  Test:       Reserved for final evaluation (use test.py)")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # ==================== MODEL ====================
    print(f"\n{'='*70}")
    print("MODEL INITIALIZATION")
    print(f"{'='*70}")
    model = JEPA_ReasoningModel(config)
    
    # ==================== TRAINER ====================
    print("Initializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config['resume_from'] is not None and os.path.exists(config['resume_from']):
        print(f"\nResuming from checkpoint: {config['resume_from']}")
        start_epoch = trainer.load_checkpoint(config['resume_from']) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # ==================== TRAIN ====================
    history = trainer.train(
        num_epochs=config['epochs'],
        save_dir=config['save_dir']
    )
    
    print("\n{'='*70}")
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\n⚠️  REMEMBER: Test set is still untouched!")
    print(f"   Run 'python test.py' for final evaluation on test set.")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()