"""
test.py - Testing and Evaluation for JEPA Reasoning Model
Includes evaluation metrics, visualization, and analysis tools
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our modules
from Encoders import JEPA_Encoders
from predictor import PredictorWithLoss, EvaluationMetrics
from train import JEPA_ReasoningModel, GSM8KDataset


class ModelTester:
    """
    Comprehensive testing and evaluation for JEPA Reasoning Model
    """
    
    def __init__(self, model, test_loader, tokenizer, config, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.metrics = EvaluationMetrics()
        
        print(f"\nTester initialized:")
        print(f"  Device: {device}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
    
    def evaluate(self, save_predictions=True, output_dir='./test_results'):
        """
        Evaluate model on test set
        
        Args:
            save_predictions: Whether to save detailed predictions
            output_dir: Directory to save results
        
        Returns:
            results: Dictionary with evaluation metrics
        """
        self.model.eval()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"EVALUATING MODEL ON TEST SET")
        print(f"{'='*70}\n")
        
        # Storage for results
        all_predictions = []
        all_losses = []
        
        # Accumulated metrics
        total_cos_sim = 0
        total_angular_dist = 0
        total_l2_dist = 0
        
        accuracy_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
        threshold_counts = {t: 0 for t in accuracy_thresholds}
        
        num_samples = 0
        
        pbar = tqdm(self.test_loader, desc="Testing")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                batch_device = {k: v.to(self.device) if torch.is_tensor(v) else v 
                               for k, v in batch.items()}
                
                # Forward pass
                predicted_emb, target_emb, embeddings_dict = self.model(
                    batch_device, 
                    return_embeddings=True
                )
                
                # Compute loss
                loss, loss_dict = self.model.compute_loss(predicted_emb, target_emb)
                all_losses.append(loss.item())
                
                # Compute metrics for this batch
                batch_metrics = self.metrics.compute_all_metrics(
                    predicted_emb, target_emb
                )
                
                total_cos_sim += batch_metrics['cosine_similarity'] * len(predicted_emb)
                total_angular_dist += batch_metrics['angular_distance'] * len(predicted_emb)
                total_l2_dist += batch_metrics['l2_distance'] * len(predicted_emb)
                
                # Count accuracies at different thresholds
                cos_sims = F.cosine_similarity(
                    F.normalize(predicted_emb, p=2, dim=-1),
                    F.normalize(target_emb, p=2, dim=-1),
                    dim=-1
                )
                
                for threshold in accuracy_thresholds:
                    threshold_counts[threshold] += (cos_sims >= threshold).sum().item()
                
                num_samples += len(predicted_emb)
                
                # Save individual predictions
                if save_predictions:
                    for i in range(len(predicted_emb)):
                        pred_dict = {
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'question': batch['question_text'][i],
                            'cot': batch['cot_text'][i],
                            'answer': batch['answer_text'][i],
                            'cosine_similarity': cos_sims[i].item(),
                            'predicted_emb': predicted_emb[i].cpu().numpy().tolist(),
                            'target_emb': target_emb[i].cpu().numpy().tolist(),
                        }
                        all_predictions.append(pred_dict)
                
                # Update progress
                pbar.set_postfix({
                    'avg_cos_sim': f"{total_cos_sim / num_samples:.4f}",
                    'loss': f"{loss.item():.4f}"
                })
        
        # Compute final averaged metrics
        results = {
            'num_samples': num_samples,
            'avg_loss': np.mean(all_losses),
            'cosine_similarity': total_cos_sim / num_samples,
            'angular_distance': total_angular_dist / num_samples,
            'l2_distance': total_l2_dist / num_samples,
        }
        
        # Add accuracy at thresholds
        for threshold in accuracy_thresholds:
            accuracy = (threshold_counts[threshold] / num_samples) * 100
            results[f'accuracy@{threshold}'] = accuracy
        
        # Print results
        print(f"\n{'='*70}")
        print(f"TEST RESULTS")
        print(f"{'='*70}")
        print(f"Total samples:         {results['num_samples']}")
        print(f"Average loss:          {results['avg_loss']:.4f}")
        print(f"Cosine similarity:     {results['cosine_similarity']:.4f}")
        print(f"Angular distance:      {results['angular_distance']:.2f}°")
        print(f"L2 distance:           {results['l2_distance']:.4f}")
        print(f"{'-'*70}")
        print(f"Accuracy Thresholds:")
        for threshold in accuracy_thresholds:
            acc = results[f'accuracy@{threshold}']
            print(f"  @ {threshold:.1f}: {acc:>6.2f}%")
        print(f"{'='*70}\n")
        
        # Save results
        results_path = os.path.join(output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {results_path}")
        
        # Save predictions
        if save_predictions:
            predictions_path = os.path.join(output_dir, 'predictions.json')
            with open(predictions_path, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            print(f"✓ Predictions saved to: {predictions_path}")
            
            # Also save as CSV for easy viewing
            predictions_csv = os.path.join(output_dir, 'predictions.csv')
            pred_df = pd.DataFrame([
                {
                    'question': p['question'],
                    'answer': p['answer'],
                    'cosine_similarity': p['cosine_similarity']
                }
                for p in all_predictions
            ])
            pred_df.to_csv(predictions_csv, index=False)
            print(f"✓ Predictions CSV saved to: {predictions_csv}")
        
        return results, all_predictions
    
    def analyze_errors(self, predictions, top_k=20, output_dir='./test_results'):
        """
        Analyze worst predictions to understand failure modes
        
        Args:
            predictions: List of prediction dictionaries
            top_k: Number of worst examples to analyze
            output_dir: Directory to save analysis
        """
        print(f"\n{'='*70}")
        print(f"ERROR ANALYSIS - Top {top_k} Worst Predictions")
        print(f"{'='*70}\n")
        
        # Sort by cosine similarity (ascending - worst first)
        sorted_preds = sorted(predictions, key=lambda x: x['cosine_similarity'])
        
        worst_examples = sorted_preds[:top_k]
        
        # Analyze
        for i, example in enumerate(worst_examples, 1):
            print(f"\n{'-'*70}")
            print(f"Rank {i} - Cosine Similarity: {example['cosine_similarity']:.4f}")
            print(f"{'-'*70}")
            print(f"Question: {example['question'][:150]}...")
            print(f"CoT: {example['cot'][:150]}...")
            print(f"Answer: {example['answer']}")
            print()
        
        # Save analysis
        analysis_path = os.path.join(output_dir, 'error_analysis.txt')
        with open(analysis_path, 'w') as f:
            f.write(f"ERROR ANALYSIS - Top {top_k} Worst Predictions\n")
            f.write(f"{'='*70}\n\n")
            
            for i, example in enumerate(worst_examples, 1):
                f.write(f"\nRank {i} - Cosine Similarity: {example['cosine_similarity']:.4f}\n")
                f.write(f"{'-'*70}\n")
                f.write(f"Question: {example['question']}\n\n")
                f.write(f"CoT: {example['cot']}\n\n")
                f.write(f"Answer: {example['answer']}\n\n")
        
        print(f"✓ Error analysis saved to: {analysis_path}")
    
    def analyze_best_predictions(self, predictions, top_k=20, output_dir='./test_results'):
        """
        Analyze best predictions to understand what the model learns well
        """
        print(f"\n{'='*70}")
        print(f"BEST PREDICTIONS ANALYSIS - Top {top_k}")
        print(f"{'='*70}\n")
        
        # Sort by cosine similarity (descending - best first)
        sorted_preds = sorted(predictions, key=lambda x: x['cosine_similarity'], reverse=True)
        
        best_examples = sorted_preds[:top_k]
        
        for i, example in enumerate(best_examples, 1):
            print(f"\n{'-'*70}")
            print(f"Rank {i} - Cosine Similarity: {example['cosine_similarity']:.4f}")
            print(f"{'-'*70}")
            print(f"Question: {example['question'][:150]}...")
            print(f"Answer: {example['answer']}")
            print()
        
        # Save analysis
        analysis_path = os.path.join(output_dir, 'best_predictions.txt')
        with open(analysis_path, 'w') as f:
            f.write(f"BEST PREDICTIONS - Top {top_k}\n")
            f.write(f"{'='*70}\n\n")
            
            for i, example in enumerate(best_examples, 1):
                f.write(f"\nRank {i} - Cosine Similarity: {example['cosine_similarity']:.4f}\n")
                f.write(f"{'-'*70}\n")
                f.write(f"Question: {example['question']}\n\n")
                f.write(f"CoT: {example['cot']}\n\n")
                f.write(f"Answer: {example['answer']}\n\n")
        
        print(f"✓ Best predictions analysis saved to: {analysis_path}")
    
    def visualize_results(self, predictions, output_dir='./test_results'):
        """
        Create visualizations of test results
        """
        print(f"\n{'='*70}")
        print(f"CREATING VISUALIZATIONS")
        print(f"{'='*70}\n")
        
        # Extract cosine similarities
        cos_sims = [p['cosine_similarity'] for p in predictions]
        
        # Set style
        sns.set_style("darkgrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('JEPA Reasoning Model - Test Results', fontsize=16, fontweight='bold')
        
        # 1. Distribution of cosine similarities
        ax1 = axes[0, 0]
        ax1.hist(cos_sims, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(cos_sims), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(cos_sims):.4f}')
        ax1.axvline(np.median(cos_sims), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(cos_sims):.4f}')
        ax1.set_xlabel('Cosine Similarity', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Cosine Similarities', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative distribution
        ax2 = axes[0, 1]
        sorted_cos_sims = np.sort(cos_sims)
        cumulative = np.arange(1, len(sorted_cos_sims) + 1) / len(sorted_cos_sims) * 100
        ax2.plot(sorted_cos_sims, cumulative, linewidth=2, color='darkblue')
        ax2.set_xlabel('Cosine Similarity', fontsize=12)
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax2.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add percentile lines
        for percentile in [25, 50, 75, 90]:
            value = np.percentile(cos_sims, percentile)
            ax2.axvline(value, color='red', linestyle=':', alpha=0.5)
            ax2.text(value, 5, f'{percentile}th', rotation=90, fontsize=9)
        
        # 3. Box plot with accuracy thresholds
        ax3 = axes[1, 0]
        bp = ax3.boxplot([cos_sims], vert=True, patch_artist=True, 
                         labels=['Cosine Similarity'])
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_edgecolor('darkblue')
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        
        # Add threshold lines
        thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
        colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen']
        for threshold, color in zip(thresholds, colors):
            ax3.axhline(threshold, color=color, linestyle='--', linewidth=1.5, 
                       label=f'Threshold: {threshold}')
            accuracy = (np.array(cos_sims) >= threshold).mean() * 100
            ax3.text(1.15, threshold, f'{accuracy:.1f}%', fontsize=9, va='center')
        
        ax3.set_ylabel('Cosine Similarity', fontsize=12)
        ax3.set_title('Box Plot with Accuracy Thresholds', fontsize=14, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        SUMMARY STATISTICS
        {'='*40}
        
        Total Samples:       {len(cos_sims)}
        
        Mean:                {np.mean(cos_sims):.4f}
        Median:              {np.median(cos_sims):.4f}
        Std Dev:             {np.std(cos_sims):.4f}
        
        Min:                 {np.min(cos_sims):.4f}
        Max:                 {np.max(cos_sims):.4f}
        
        25th Percentile:     {np.percentile(cos_sims, 25):.4f}
        75th Percentile:     {np.percentile(cos_sims, 75):.4f}
        90th Percentile:     {np.percentile(cos_sims, 90):.4f}
        
        ACCURACY THRESHOLDS
        {'-'*40}
        """
        
        for threshold in thresholds:
            accuracy = (np.array(cos_sims) >= threshold).mean() * 100
            stats_text += f"\n        @ {threshold:.1f}:  {accuracy:>6.2f}%"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, 'test_results_visualization.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {fig_path}")
        
        plt.close()
        
        # Additional plot: Scatter plot of predictions
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sample predictions for visualization (to avoid clutter)
        sample_size = min(1000, len(predictions))
        sampled_preds = np.random.choice(predictions, sample_size, replace=False)
        
        indices = np.arange(len(sampled_preds))
        cos_sims_sampled = [p['cosine_similarity'] for p in sampled_preds]
        
        scatter = ax.scatter(indices, cos_sims_sampled, 
                           c=cos_sims_sampled, cmap='RdYlGn', 
                           s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.axhline(0.9, color='darkgreen', linestyle='--', linewidth=2, label='Excellent (0.9)')
        ax.axhline(0.7, color='orange', linestyle='--', linewidth=2, label='Good (0.7)')
        ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Poor (0.5)')
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(f'Prediction Quality Scatter Plot (n={sample_size})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=11)
        
        scatter_path = os.path.join(output_dir, 'prediction_scatter.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"✓ Scatter plot saved to: {scatter_path}")
        
        plt.close()


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        config: Model configuration
    """
    print(f"\nLoading model from checkpoint: {checkpoint_path}")
    
    # Load with weights_only=False for backward compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Initialize model
    model = JEPA_ReasoningModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best val cosine similarity: {checkpoint.get('best_val_similarity', 'N/A')}")
    
    return model, config

def main():
    """
    Main testing script
    """
    
    # ==================== CONFIGURATION ====================
    checkpoint_path = './checkpoints/best_model.pt'
    test_csv = './gsm8k_test.csv'
    output_dir = './test_results'
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"JEPA REASONING MODEL - TESTING")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data: {test_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # ==================== LOAD MODEL ====================
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    
    # ==================== TOKENIZER ====================
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")
    
    # ==================== TEST DATASET ====================
    print("\nPreparing test dataset...")
    test_dataset = GSM8KDataset(
        test_csv,
        tokenizer,
        config,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Test dataset prepared: {len(test_dataset)} examples")
    
    # ==================== TESTER ====================
    tester = ModelTester(model, test_loader, tokenizer, config, device)
    
    # ==================== EVALUATE ====================
    results, predictions = tester.evaluate(
        save_predictions=True,
        output_dir=output_dir
    )
    
    # ==================== ERROR ANALYSIS ====================
    tester.analyze_errors(predictions, top_k=20, output_dir=output_dir)
    tester.analyze_best_predictions(predictions, top_k=20, output_dir=output_dir)
    
    # ==================== VISUALIZATIONS ====================
    tester.visualize_results(predictions, output_dir=output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ TESTING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()