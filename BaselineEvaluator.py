import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
import pandas as pd
import json
import re
from tqdm import tqdm
from pathlib import Path
import time

class PretrainedBaselineEvaluator:
    """
    Evaluate pre-trained models on GSM8K without fine-tuning
    """
    
    def __init__(self, model_name, model_type="causal"):
        """
        Initialize model for evaluation
        
        Args:
            model_name: HuggingFace model ID
            model_type: "causal" (GPT-2, Llama, etc.) or "seq2seq" (T5, FLAN-T5)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n{'='*80}")
        print(f"Loading {model_name}")
        print(f"{'='*80}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"✓ Model: {model_name}")
        print(f"✓ Type: {model_type}")
        print(f"✓ Parameters: {num_params:.1f}M")
        print(f"✓ Device: {self.device}")
        
    def load_data(self, test_csv):
        """Load test data"""
        self.test_df = pd.read_csv(test_csv)
        print(f"✓ Loaded {len(self.test_df)} test examples")
        
    def create_zero_shot_prompt(self, question):
        """Create zero-shot prompt"""
        if self.model_type == "causal":
            return f"""Question: {question}

Let's solve this step by step:
"""
        else:  # seq2seq
            return f"Solve this math problem step by step: {question}"
    
    def create_few_shot_prompt(self, question, num_shots=3):
        """Create few-shot prompt with examples"""
        # Few-shot examples from GSM8K training set
        examples = [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "solution": "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\nShe makes 9 * 2 = 18 every day at the farmer's market.\nThe answer is 18",
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "solution": "It takes 2/2=1 bolt of white fiber\nSo the total amount of fabric is 2+1=3 bolts of fiber\nThe answer is 3",
            },
            {
                "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                "solution": "The cost of the house and repairs came out to 80,000+50,000=$130,000\nHe increased the value of the house by 80,000*1.5=120,000\nSo the new value of the house is 120,000+80,000=$200,000\nSo he made a profit of 200,000-130,000=$70,000\nThe answer is 70000",
            }
        ]
        
        # Select examples
        selected_examples = examples[:num_shots]
        
        if self.model_type == "causal":
            prompt = ""
            for ex in selected_examples:
                prompt += f"Question: {ex['question']}\n\n"
                prompt += f"Solution: {ex['solution']}\n\n"
                prompt += "---\n\n"
            
            prompt += f"Question: {question}\n\n"
            prompt += "Solution:"
            
        else:  # seq2seq
            prompt = "Solve these math problems step by step:\n\n"
            for ex in selected_examples:
                prompt += f"Q: {ex['question']}\nA: {ex['solution']}\n\n"
            prompt += f"Q: {question}\nA:"
        
        return prompt
    
    def extract_answer(self, text):
        """Extract numerical answer from generated text"""
        text = text.lower()
        
        # Common patterns
        patterns = [
            r'the answer is[:\s]+([+-]?[0-9,]+\.?[0-9]*)',
            r'answer[:\s]+([+-]?[0-9,]+\.?[0-9]*)',
            r'####\s*([+-]?[0-9,]+\.?[0-9]*)',
            r'=\s*\$?\s*([+-]?[0-9,]+\.?[0-9]*)\s*$',
            r'\$?\s*([+-]?[0-9,]+\.?[0-9]*)\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).replace(',', '').strip()
        
        # Fallback: last number in text
        numbers = re.findall(r'([+-]?[0-9,]+\.?[0-9]*)', text)
        if numbers:
            return numbers[-1].replace(',', '').strip()
        
        return ""
    
    def normalize_answer(self, answer):
        """Normalize answer for comparison"""
        answer = str(answer).strip().replace(',', '')
        try:
            # Handle floats and ints
            num = float(answer)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except:
            return answer
    
    def generate_answer(self, prompt, max_new_tokens=256):
        """Generate answer for a prompt"""
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else None,
                eos_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None,
            )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # For causal models, remove the prompt
            if self.model_type == "causal":
                generated = generated[len(prompt):].strip()
            
            return generated
    
    def evaluate(self, mode="zero-shot", num_shots=3, num_samples=None, save_results=True):
        """
        Evaluate model on test set
        
        Args:
            mode: "zero-shot" or "few-shot"
            num_shots: Number of examples for few-shot (if mode="few-shot")
            num_samples: Number of test samples (None = all)
            save_results: Whether to save detailed results
        """
        print(f"\n{'='*80}")
        print(f"EVALUATION - {mode.upper()}")
        if mode == "few-shot":
            print(f"Number of shots: {num_shots}")
        print(f"{'='*80}")
        
        test_df = self.test_df if num_samples is None else self.test_df.head(num_samples)
        
        correct = 0
        total = 0
        results = []
        
        start_time = time.time()
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            question = row['question']
            ground_truth = self.normalize_answer(row['answer'])
            
            # Create prompt
            if mode == "zero-shot":
                prompt = self.create_zero_shot_prompt(question)
            elif mode == "few-shot":
                prompt = self.create_few_shot_prompt(question, num_shots)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Generate answer
            try:
                generated = self.generate_answer(prompt)
                predicted = self.normalize_answer(self.extract_answer(generated))
            except Exception as e:
                print(f"\nError on example {idx}: {e}")
                predicted = ""
                generated = ""
            
            # Check correctness
            is_correct = predicted == ground_truth
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            results.append({
                'idx': idx,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'correct': is_correct,
                'generated_text': generated[:300]  # First 300 chars
            })
        
        elapsed_time = time.time() - start_time
        accuracy = (correct / total) * 100
        
        print(f"\n{'='*80}")
        print(f"RESULTS - {mode.upper()}")
        print(f"{'='*80}")
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Time: {elapsed_time:.1f}s ({elapsed_time/total:.2f}s per example)")
        print(f"{'='*80}")
        
        # Show a few examples
        print(f"\nSample results:")
        print(f"{'-'*80}")
        for i, result in enumerate(results[:3]):
            status = "✓" if result['correct'] else "✗"
            print(f"\n{status} Example {i+1}:")
            print(f"Question: {result['question'][:80]}...")
            print(f"Ground truth: {result['ground_truth']}")
            print(f"Predicted: {result['predicted']}")
            print(f"Generated: {result['generated_text'][:100]}...")
        
        # Save results
        if save_results:
            output_file = f"results_{self.model_name.replace('/', '_')}_{mode}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'model': self.model_name,
                    'mode': mode,
                    'num_shots': num_shots if mode == "few-shot" else 0,
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'elapsed_time': elapsed_time,
                    'detailed_results': results
                }, f, indent=2)
            print(f"\n✓ Results saved to {output_file}")
        
        return accuracy, results


def run_pretrained_baselines(test_csv, output_dir="./pretrained_baseline_results"):
    """
    Run comprehensive pre-trained model baselines
    
    Args:
        test_csv: Path to test CSV
        output_dir: Directory to save results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Models to test
    models = [
        {
            "name": "GPT-2 Small",
            "model_id": "gpt2",
            "type": "causal"
        },
        {
            "name": "GPT-2 Medium",
            "model_id": "gpt2-medium",
            "type": "causal"
        },
        {
            "name": "T5-Small",
            "model_id": "t5-small",
            "type": "seq2seq"
        },
        {
            "name": "FLAN-T5-Small",
            "model_id": "google/flan-t5-small",
            "type": "seq2seq"
        }
    ]
    
    all_results = {}
    
    for model_config in models:
        model_name = model_config["name"]
        model_id = model_config["model_id"]
        model_type = model_config["type"]
        
        print(f"\n\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*80}\n")
        
        try:
            # Initialize evaluator
            evaluator = PretrainedBaselineEvaluator(model_id, model_type)
            evaluator.load_data(test_csv)
            
            # Zero-shot evaluation
            print(f"\n>>> Running zero-shot evaluation...")
            zero_shot_acc, _ = evaluator.evaluate(
                mode="zero-shot",
                save_results=True
            )
            
            # Few-shot evaluation (3-shot)
            print(f"\n>>> Running few-shot evaluation (3-shot)...")
            few_shot_3_acc, _ = evaluator.evaluate(
                mode="few-shot",
                num_shots=3,
                save_results=True
            )
            
            # Few-shot evaluation (8-shot)
            print(f"\n>>> Running few-shot evaluation (8-shot)...")
            few_shot_8_acc, _ = evaluator.evaluate(
                mode="few-shot",
                num_shots=8,
                save_results=True
            )
            
            # Store results
            all_results[model_name] = {
                "model_id": model_id,
                "type": model_type,
                "zero_shot": zero_shot_acc,
                "few_shot_3": few_shot_3_acc,
                "few_shot_8": few_shot_8_acc
            }
            
        except Exception as e:
            print(f"\n✗ Error with {model_name}: {e}")
            all_results[model_name] = {
                "model_id": model_id,
                "type": model_type,
                "error": str(e)
            }
    
    # Print summary
    print(f"\n\n{'#'*80}")
    print(f"# BASELINE RESULTS SUMMARY")
    print(f"{'#'*80}\n")
    
    print(f"{'Model':<20} {'Zero-shot':<12} {'3-shot':<12} {'8-shot':<12}")
    print(f"{'-'*80}")
    
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"{model_name:<20} ERROR: {results['error']}")
        else:
            print(f"{model_name:<20} {results['zero_shot']:>6.2f}%     "
                  f"{results['few_shot_3']:>6.2f}%     "
                  f"{results['few_shot_8']:>6.2f}%")
    
    # Save summary
    summary_file = f"{output_dir}/pretrained_baseline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_file}")
    
    return all_results


def main():
    """Main function"""
    # ===== CONFIGURE =====
    TEST_CSV = "R-JEPA/gsm8k_test.csv"
    OUTPUT_DIR = "./pretrained_baseline_results"
    # =====================
    
    print(f"\n{'='*80}")
    print(f"PRE-TRAINED MODEL BASELINE EVALUATION")
    print(f"{'='*80}")
    print(f"Test data: {TEST_CSV}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*80}\n")
    
    # Run baselines
    results = run_pretrained_baselines(TEST_CSV, OUTPUT_DIR)
    
    print("\n✅ All evaluations complete!")


if __name__ == "__main__":
    main()ð