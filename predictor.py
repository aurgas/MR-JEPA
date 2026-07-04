"""
predictor.py - JEPA Reasoning Model Predictor with Cosine Similarity
Implements cross-attention predictor that combines question and CoT embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionPredictor(nn.Module):
    """
    Cross-Attention Predictor for JEPA Reasoning Model
    
    Question embedding attends to CoT embedding to extract relevant reasoning,
    then processes through FFN to predict answer embedding.
    
    Output is L2-normalized for cosine similarity loss.
    """
    
    def __init__(self, d_model, n_heads=4, d_ff=None, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension (512)
            n_heads: Number of attention heads (4)
            d_ff: Feed-forward hidden dimension (default: 2 * d_model)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff if d_ff is not None else 2 * d_model
        
        # Cross-attention: question queries CoT
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Final projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._print_model_info()
    
    def _print_model_info(self):
        """Print predictor architecture info"""
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        
        print(f"\n{'='*70}")
        print(f"Cross-Attention Predictor")
        print(f"{'='*70}")
        print(f"Input dimension:     {self.d_model}")
        print(f"Attention heads:     {self.n_heads}")
        print(f"FFN hidden dim:      {self.d_ff}")
        print(f"Output dimension:    {self.d_model}")
        print(f"Total parameters:    {total_params:.1f}M")
        print(f"Loss function:       Cosine Similarity")
        print(f"{'='*70}\n")
    
    def forward(self, question_emb, cot_emb, normalize=True):
        """
        Forward pass with cross-attention
        
        Args:
            question_emb: [batch_size, d_model] - from question encoder
            cot_emb: [batch_size, d_model] - from CoT encoder
            normalize: Whether to L2 normalize output (for cosine similarity)
        
        Returns:
            predicted_emb: [batch_size, d_model] - predicted answer embedding
        """
        # Add sequence dimension for attention
        # [B, d_model] -> [B, 1, d_model]
        query = question_emb.unsqueeze(1)
        key = cot_emb.unsqueeze(1)
        value = cot_emb.unsqueeze(1)
        
        # Cross-attention: question attends to CoT
        attn_output, attn_weights = self.cross_attention(
            query, key, value
        )  # [B, 1, d_model]
        
        # Remove sequence dimension
        attn_output = attn_output.squeeze(1)  # [B, d_model]
        
        # Residual connection and normalization
        x = self.norm1(question_emb + self.dropout(attn_output))
        
        # Feed-forward network with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        # Final projection
        predicted_emb = self.output_projection(x)  # [B, d_model]
        
        # L2 normalize for cosine similarity
        if normalize:
            predicted_emb = F.normalize(predicted_emb, p=2, dim=-1)
        
        return predicted_emb


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss for latent space prediction
    
    Loss = 1 - cosine_similarity(predicted, target)
    Range: [0, 2] where 0 is perfect alignment
    """
    
    def __init__(self, temperature=1.0):
        """
        Args:
            temperature: Temperature for scaling (default: 1.0)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predicted_emb, target_emb):
        """
        Compute cosine similarity loss
        
        Args:
            predicted_emb: [batch_size, d_model] - predicted embeddings
            target_emb: [batch_size, d_model] - target embeddings
        
        Returns:
            loss: scalar tensor
        """
        # Normalize both embeddings
        predicted_emb = F.normalize(predicted_emb, p=2, dim=-1)
        target_emb = F.normalize(target_emb, p=2, dim=-1)
        
        # Cosine similarity: ranges from -1 (opposite) to 1 (same)
        cos_sim = F.cosine_similarity(predicted_emb, target_emb, dim=-1)
        
        # Apply temperature scaling
        cos_sim = cos_sim / self.temperature
        
        # Convert to loss: want similarity = 1, so loss = 1 - similarity
        loss = (1 - cos_sim).mean()
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (InfoNCE) for latent space prediction
    
    Uses batch as negative examples to encourage discriminative embeddings.
    More powerful than simple cosine similarity loss.
    """
    
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Temperature for scaling similarities (default: 0.07)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predicted_emb, target_emb):
        """
        Compute contrastive loss
        
        Args:
            predicted_emb: [batch_size, d_model]
            target_emb: [batch_size, d_model]
        
        Returns:
            loss: scalar tensor
        """
        batch_size = predicted_emb.shape[0]
        
        # Normalize embeddings
        predicted_emb = F.normalize(predicted_emb, p=2, dim=-1)
        target_emb = F.normalize(target_emb, p=2, dim=-1)
        
        # Compute similarity matrix: [batch_size, batch_size]
        # similarity[i, j] = cosine similarity between predicted[i] and target[j]
        similarity_matrix = torch.matmul(
            predicted_emb, target_emb.T
        ) / self.temperature
        
        # Labels: diagonal elements are positive pairs (i -> i)
        labels = torch.arange(batch_size, device=predicted_emb.device)
        
        # Cross-entropy loss
        # For each predicted[i], target[i] should have highest similarity
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining cosine similarity and contrastive loss
    
    Balances direct alignment (cosine) with discriminative learning (contrastive)
    """
    
    def __init__(self, alpha=0.7, temperature=0.07):
        """
        Args:
            alpha: Weight for cosine loss (1-alpha for contrastive)
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.alpha = alpha
        self.cosine_loss = CosineSimilarityLoss(temperature=1.0)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
    
    def forward(self, predicted_emb, target_emb):
        """
        Compute hybrid loss
        
        Args:
            predicted_emb: [batch_size, d_model]
            target_emb: [batch_size, d_model]
        
        Returns:
            loss: scalar tensor
            loss_dict: dictionary with individual losses
        """
        # Compute both losses
        cos_loss = self.cosine_loss(predicted_emb, target_emb)
        con_loss = self.contrastive_loss(predicted_emb, target_emb)
        
        # Weighted combination
        total_loss = self.alpha * cos_loss + (1 - self.alpha) * con_loss
        
        # Return total and individual losses for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'cosine_loss': cos_loss.item(),
            'contrastive_loss': con_loss.item()
        }
        
        return total_loss, loss_dict


class PredictorWithLoss(nn.Module):
    """
    Complete predictor module with integrated loss function
    Combines cross-attention predictor with cosine similarity loss
    """
    
    def __init__(self, config, loss_type='cosine'):
        """
        Args:
            config: Model configuration dictionary
            loss_type: 'cosine', 'contrastive', or 'hybrid'
        """
        super().__init__()
        
        self.config = config
        self.loss_type = loss_type
        
        # Predictor
        self.predictor = CrossAttentionPredictor(
            d_model=config['d_model'],
            n_heads=config.get('predictor_heads', 4),
            d_ff=config.get('predictor_d_ff', config['d_model'] * 2),
            dropout=config['dropout']
        )
        
        # Loss function
        if loss_type == 'cosine':
            self.loss_fn = CosineSimilarityLoss(
                temperature=config.get('temperature', 1.0)
            )
        elif loss_type == 'contrastive':
            self.loss_fn = ContrastiveLoss(
                temperature=config.get('temperature', 0.07)
            )
        elif loss_type == 'hybrid':
            self.loss_fn = HybridLoss(
                alpha=config.get('hybrid_alpha', 0.7),
                temperature=config.get('temperature', 0.07)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        print(f"Initialized predictor with loss type: {loss_type}")
    
    def forward(self, question_emb, cot_emb):
        """
        Forward pass through predictor
        
        Args:
            question_emb: [batch_size, d_model]
            cot_emb: [batch_size, d_model]
        
        Returns:
            predicted_emb: [batch_size, d_model] - L2 normalized
        """
        return self.predictor(question_emb, cot_emb, normalize=True)
    
    def compute_loss(self, predicted_emb, target_emb):
        """
        Compute loss between predicted and target embeddings
        
        Args:
            predicted_emb: [batch_size, d_model]
            target_emb: [batch_size, d_model]
        
        Returns:
            loss: scalar tensor (and loss_dict for hybrid loss)
        """
        if self.loss_type == 'hybrid':
            return self.loss_fn(predicted_emb, target_emb)
        else:
            loss = self.loss_fn(predicted_emb, target_emb)
            return loss, {'total_loss': loss.item()}


class EvaluationMetrics:
    """
    Evaluation metrics for predictor performance
    All metrics work with normalized embeddings
    """
    
    @staticmethod
    def cosine_similarity(predicted_emb, target_emb):
        """
        Average cosine similarity across batch
        
        Returns:
            similarity: float in [-1, 1], higher is better
        """
        predicted_emb = F.normalize(predicted_emb, p=2, dim=-1)
        target_emb = F.normalize(target_emb, p=2, dim=-1)
        
        cos_sim = F.cosine_similarity(predicted_emb, target_emb, dim=-1)
        return cos_sim.mean().item()
    
    @staticmethod
    def angular_distance(predicted_emb, target_emb):
        """
        Angular distance in degrees
        
        Returns:
            distance: float in [0, 180], lower is better
        """
        predicted_emb = F.normalize(predicted_emb, p=2, dim=-1)
        target_emb = F.normalize(target_emb, p=2, dim=-1)
        
        cos_sim = F.cosine_similarity(predicted_emb, target_emb, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # Numerical stability
        
        angle_rad = torch.acos(cos_sim)
        angle_deg = torch.rad2deg(angle_rad)
        
        return angle_deg.mean().item()
    
    @staticmethod
    def l2_distance(predicted_emb, target_emb):
        """
        Euclidean distance between embeddings
        
        Returns:
            distance: float, lower is better
        """
        return F.pairwise_distance(predicted_emb, target_emb, p=2).mean().item()
    
    @staticmethod
    def accuracy_at_threshold(predicted_emb, target_emb, threshold=0.9):
        """
        Percentage of predictions within cosine similarity threshold
        
        Args:
            threshold: Minimum cosine similarity to be "correct"
        
        Returns:
            accuracy: float in [0, 100]
        """
        predicted_emb = F.normalize(predicted_emb, p=2, dim=-1)
        target_emb = F.normalize(target_emb, p=2, dim=-1)
        
        cos_sim = F.cosine_similarity(predicted_emb, target_emb, dim=-1)
        correct = (cos_sim >= threshold).float()
        
        return correct.mean().item() * 100
    
    @staticmethod
    def compute_all_metrics(predicted_emb, target_emb):
        """
        Compute all evaluation metrics
        
        Returns:
            metrics: dictionary with all metrics
        """
        return {
            'cosine_similarity': EvaluationMetrics.cosine_similarity(
                predicted_emb, target_emb
            ),
            'angular_distance': EvaluationMetrics.angular_distance(
                predicted_emb, target_emb
            ),
            'l2_distance': EvaluationMetrics.l2_distance(
                predicted_emb, target_emb
            ),
            'accuracy@0.9': EvaluationMetrics.accuracy_at_threshold(
                predicted_emb, target_emb, threshold=0.9
            ),
            'accuracy@0.8': EvaluationMetrics.accuracy_at_threshold(
                predicted_emb, target_emb, threshold=0.8
            ),
            'accuracy@0.7': EvaluationMetrics.accuracy_at_threshold(
                predicted_emb, target_emb, threshold=0.7
            )
        }


def test_predictor():
    """
    Test script for predictor module
    """
    print("Testing Cross-Attention Predictor with Cosine Similarity Loss...")
    
    # Configuration
    config = {
        'd_model': 512,
        'predictor_heads': 4,
        'predictor_d_ff': 1024,
        'dropout': 0.1,
        'temperature': 0.07,
        'hybrid_alpha': 0.7
    }
    
    # Test all loss types
    for loss_type in ['cosine', 'contrastive', 'hybrid']:
        print(f"\n{'='*70}")
        print(f"Testing with {loss_type.upper()} loss")
        print(f"{'='*70}")
        
        # Initialize predictor with loss
        predictor_with_loss = PredictorWithLoss(config, loss_type=loss_type)
        predictor_with_loss.eval()
        
        # Create dummy inputs
        batch_size = 8
        d_model = config['d_model']
        
        question_emb = torch.randn(batch_size, d_model)
        cot_emb = torch.randn(batch_size, d_model)
        target_emb = torch.randn(batch_size, d_model)
        
        # Normalize target (simulating Y-encoder output)
        target_emb = F.normalize(target_emb, p=2, dim=-1)
        
        # Forward pass
        with torch.no_grad():
            predicted_emb = predictor_with_loss(question_emb, cot_emb)
            
            # Compute loss
            if loss_type == 'hybrid':
                loss, loss_dict = predictor_with_loss.compute_loss(
                    predicted_emb, target_emb
                )
            else:
                loss, loss_dict = predictor_with_loss.compute_loss(
                    predicted_emb, target_emb
                )
        
        # Print results
        print(f"\nOutput shapes:")
        print(f"  Predicted embedding: {predicted_emb.shape}")
        print(f"  Target embedding:    {target_emb.shape}")
        
        print(f"\nLoss values:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.4f}")
        
        # Check normalization
        pred_norm = torch.norm(predicted_emb, p=2, dim=-1).mean().item()
        target_norm = torch.norm(target_emb, p=2, dim=-1).mean().item()
        
        print(f"\nNormalization check:")
        print(f"  Predicted L2 norm: {pred_norm:.4f} (should be ~1.0)")
        print(f"  Target L2 norm:    {target_norm:.4f} (should be ~1.0)")
        
        # Compute evaluation metrics
        metrics = EvaluationMetrics.compute_all_metrics(predicted_emb, target_emb)
        
        print(f"\nEvaluation metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    print(f"\n{'='*70}")
    print("✓ All tests passed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_predictor()