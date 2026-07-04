"""
test_pipeline.py - Tests the End-to-End R-JEPA capabilities.
Text (Question + CoT) -> JEPA Predictor (Latent Emb) -> Decoder -> Generated Text (Answer)
"""
import torch
from transformers import AutoTokenizer
import os
import pandas as pd
import re
from tqdm import tqdm

from train import JEPA_ReasoningModel
from decoder import JEPADecoder


def test_end_to_end_pipeline():
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
        'decoder_layers': 2,
        
        # Testing Configs
        'test_csv': 'gsm8k_test.csv',
        'jepa_checkpoint': './checkpoints/best_model.pt',
        'decoder_checkpoint': './checkpoints_decoder/best_decoder.pt',
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Testing End-to-End Pipeline on device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    config['vocab_size'] = len(tokenizer)
    
    print("\nLoading Models...")
    jepa_model = JEPA_ReasoningModel(config).to(device)
    decoder = JEPADecoder(config).to(device)
    
    # 1. Load Phase 1 Checkpoint
    if os.path.exists(config['jepa_checkpoint']):
        jepa_ckpt = torch.load(config['jepa_checkpoint'], map_location=device, weights_only=False)
        jepa_model.load_state_dict(jepa_ckpt['model_state_dict'])
        print(f"✓ Formatted Phase 1 JEPA model loaded from {config['jepa_checkpoint']}")
    else:
        print(f"⚠️ Phase 1 JEPA model missing at {config['jepa_checkpoint']}")
        
    # 2. Load Phase 2 Checkpoint
    if os.path.exists(config['decoder_checkpoint']):
        dec_ckpt = torch.load(config['decoder_checkpoint'], map_location=device, weights_only=False)
        decoder.load_state_dict(dec_ckpt['decoder_state_dict'])
        print(f"✓ Formatted Phase 2 Decoder loaded from {config['decoder_checkpoint']}")
    else:
        print(f"⚠️ Phase 2 Decoder missing at {config['decoder_checkpoint']}")
        
    jepa_model.eval()
    decoder.eval()
    
    # 3. Load Testing Data
    if not os.path.exists(config['test_csv']):
        print(f"\nTest set {config['test_csv']} missing - Aborting execution.")
        return
        
    df = pd.read_csv(config['test_csv'])
    print(f"\nSuccessfully loaded {len(df)} sample rows from testing set. Generating examples:")
    
    # 4. Initialize Evaluation Metrics and Output File
    correct = 0
    incorrect = 0
    results_file = "end2end_test_results.txt"
    
    def extract_number(text):
        if not text:
            return None
        # Try finding the number after #### (standard GSM8K format)
        match = re.search(r'####\s*(-?\d+\.?\d*)', text)
        if match:
            return match.group(1)
        # Fallback to the last number in the text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]
        return None
        
    print(f"Starting evaluation on {len(df)} samples... saving to {results_file}")
    
    # 5. Generate Output Stream and Evaluate Accuracy
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"R-JEPA End-to-End Test Results - {len(df)} samples\n")
        f.write("="*70 + "\n\n")
        
        with torch.no_grad():
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Models"):
                question = str(row['question'])
                cot = str(row['cot'])
                true_answer = str(row['answer'])
                
                # Tokenize Inputs
                q_enc = tokenizer(question, max_length=config['max_seq_len_question'], padding='max_length', truncation=True, return_tensors='pt').to(device)
                c_enc = tokenizer(cot, max_length=config['max_seq_len_cot'], padding='max_length', truncation=True, return_tensors='pt').to(device)
                
                # Phase 1: Forward JEPA
                q_emb, c_emb, _ = jepa_model.encoders(
                    question_ids=q_enc['input_ids'],
                    cot_ids=c_enc['input_ids'],
                    answer_ids=None,
                    question_mask=q_enc['attention_mask'],
                    cot_mask=c_enc['attention_mask'],
                    answer_mask=None
                )
                
                predicted_emb = jepa_model.predictor_with_loss(q_emb, c_emb)
                
                # Phase 2: Autoregressive Decoder
                bos_token = tokenizer.eos_token_id 
                generated_ids = decoder.generate(
                    predicted_emb=predicted_emb, 
                    bos_token_id=bos_token,
                    eos_token_id=tokenizer.eos_token_id, 
                    max_new_tokens=config['max_seq_len_answer']
                )
                
                generated_str = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract numbers for exact match
                target_num = extract_number(true_answer)
                pred_num = extract_number(generated_str)
                
                # Determine Correct or Incorrect
                is_correct = (target_num is not None) and (target_num == pred_num)
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
                
                # Log to file
                f.write(f"Question: {question}\n")
                f.write(f"Ground-Truth Target Answer: {true_answer}\n")
                f.write(f"Ground-Truth Extracted Num: {target_num}\n")
                f.write(f"R-JEPA Derived Answer: {generated_str}\n")
                f.write(f"R-JEPA Extracted Num: {pred_num}\n")
                f.write(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}\n")
                f.write("-" * 50 + "\n\n")

    # 6. Calculate final accuracy and metric visualization
    total = correct + incorrect
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    
    print("\n" + "="*70)
    print("End-to-End Evaluation Test Complete.")
    print(f"Total Evaluated: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Detailed logging saved to: {results_file}")
    
    # Visualization using matplotlib
    try:
        import matplotlib.pyplot as plt
        
        labels = ['Correct', 'Incorrect']
        counts = [correct, incorrect]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, counts, color=['#4CAF50', '#F44336'])
        plt.title(f'R-JEPA Final End-to-End Accuracy: {accuracy:.2f}%')
        plt.ylabel('Number of Samples')
        
        # Add value text on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + (total * 0.01), int(yval), va='bottom', ha='center')
            
        plt.savefig('accuracy_visualization.png')
        plt.close()
        print("Accuracy bar chart visualization saved to 'accuracy_visualization.png'")
    except ImportError:
        print("Matplotlib is not installed. Skipping visualization output.")
        print("To generate visualizations, please run: pip install matplotlib")

if __name__ == "__main__":
    test_end_to_end_pipeline()
