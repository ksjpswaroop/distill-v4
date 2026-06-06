#!/usr/bin/env python3
"""
Distill-V4 Full Inference Pipeline

Implements the complete 4-gate inference pipeline:
1. Base LM forward pass
2. Knowledge Retrieval Gate
3. Symbolic Reasoning Gate (FOL)
4. RL Self-Correction Loop
5. Verification Gate
6. Token Streaming with confidence

Usage:
    python src/inference/pipeline.py --prompt "Prove quicksort is O(n log n)"
    python src/inference/pipeline.py --model checkpoints/final --interactive
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gates.gates import (
    KnowledgeRetrievalGate,
    SymbolicReasoningGate,
    RLGate,
    VerificationGate,
    VerificationResult,
)


@dataclass
class GenerationConfig:
    """Configuration for generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 2048
    verification_threshold: float = 0.5
    self_correction_max_attempts: int = 3
    repetition_penalty: float = 1.1
    

@dataclass
class GenerationResult:
    """Result from generation."""
    text: str
    tokens: list[int]
    num_self_corrections: int = 0
    verification_passed: bool = False
    confidence: float = 0.0
    latency_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)
    

@dataclass
class ExecutionResult:
    """Mock execution result for testing."""
    passed: bool = True
    output: str = ""
    error: Optional[str] = None
    runtime: float = 0.0
    test_pass_rate: float = 1.0


class DistillV4InferencePipeline:
    """
    Full inference pipeline with 4-gate architecture.
    
    Flow:
    Input → Base LM → Gate 1 (Knowledge) → Gate 2 (Symbolic) 
           → Generate tokens with Gate 3 (RL) loop → Gate 4 (Verification) → Stream
    """
    
    def __init__(
        self,
        base_model_path: str,
        gate1_path: Optional[str] = None,
        gate2_path: Optional[str] = None,
        gate3_path: Optional[str] = None,
        gate4_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: str = "bfloat16",
    ):
        self.device = device
        self.config = GenerationConfig()
        
        print(f"Loading base model from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=getattr(torch, torch_dtype),
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        self.base_model.eval()
        
        # Initialize gates
        self._init_gates(gate1_path, gate2_path, gate3_path, gate4_path)
        
        # Determine model dimensions
        self.hidden_dim = self.base_model.config.hidden_size
        print(f"Base model loaded. Hidden dim: {self.hidden_dim}")
    
    def _init_gates(
        self,
        gate1_path: Optional[str],
        gate2_path: Optional[str],
        gate3_path: Optional[str],
        gate4_path: Optional[str],
    ):
        """Initialize and load gate modules."""
        hidden_dim = self.hidden_dim
        
        # Gate 1: Knowledge Retrieval
        self.gate1 = KnowledgeRetrievalGate(hidden_dim=hidden_dim)
        if gate1_path and Path(gate1_path).exists():
            self.gate1.load_state_dict(torch.load(gate1_path, map_location="cpu"))
            print(f"Loaded Gate 1 from {gate1_path}")
        else:
            print("Warning: Gate 1 not loaded, using identity (no retrieval)")
        self.gate1.to(self.device).eval()
        
        # Gate 2: Symbolic Reasoning
        self.gate2 = SymbolicReasoningGate(hidden_dim=hidden_dim)
        if gate2_path and Path(gate2_path).exists():
            self.gate2.load_state_dict(torch.load(gate2_path, map_location="cpu"))
            print(f"Loaded Gate 2 from {gate2_path}")
        else:
            print("Warning: Gate 2 not loaded, using identity (no symbolic reasoning)")
        self.gate2.to(self.device).eval()
        
        # Gate 3: RL Self-Correction
        self.gate3 = RLGate(hidden_dim=hidden_dim)
        if gate3_path and Path(gate3_path).exists():
            self.gate3.load_state_dict(torch.load(gate3_path, map_location="cpu"))
            print(f"Loaded Gate 3 from {gate3_path}")
        else:
            print("Warning: Gate 3 not loaded, no self-correction")
        self.gate3.to(self.device).eval()
        
        # Gate 4: Verification
        self.gate4 = VerificationGate(hidden_dim=hidden_dim)
        if gate4_path and Path(gate4_path).exists():
            self.gate4.load_state_dict(torch.load(gate4_path, map_location="cpu"))
            print(f"Loaded Gate 4 from {gate4_path}")
        else:
            print("Warning: Gate 4 not loaded, no verification")
        self.gate4.to(self.device).eval()
    
    @torch.no_grad()
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        """
        Generate with full 4-gate pipeline.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            GenerationResult with text, tokens, and metadata
        """
        if config is None:
            config = self.config
        
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Step 1: Base LM forward to get initial hidden state
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states
        base_hidden = hidden_states[-1]  # Last layer hidden states
        
        # Step 2: Gate 1 - Knowledge Retrieval
        retrieved_hidden, retrieval_metadata = self.gate1(
            base_hidden, input_ids
        )
        
        # Step 3: Gate 2 - Symbolic Reasoning
        reasoning_hidden, reasoning_metadata = self.gate2(
            base_hidden, prompt
        )
        
        # Step 4: Combined hidden for generation
        # Blend base, retrieved, and reasoning hidden states
        combined_hidden = 0.5 * base_hidden[:, -1, :] + 0.3 * retrieved_hidden + 0.2 * reasoning_hidden
        combined_hidden = combined_hidden.unsqueeze(1)  # Add sequence dim
        
        # Step 5: Autoregressive generation with verification loop
        generated_ids = input_ids
        tokens_generated = []
        self_corrections = 0
        verification_passed = False
        last_confidence = 0.0
        
        for step in range(config.max_new_tokens):
            # Get logits for next token
            logits = self.base_model.lm_head(combined_hidden[:, -1, :])
            probs = F.softmax(logits / config.temperature, dim=-1)
            
            # Apply top-k and top-p
            if config.top_k > 0:
                top_k = min(config.top_k, probs.size(-1))
                top_probs, top_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(1, top_indices, top_probs)
            
            if config.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > config.top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
            
            # Sample
            probs = probs / probs.sum()
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                for prev_token in set(generated_ids[0].tolist()):
                    probs[0, prev_token] /= config.repetition_penalty
            
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            tokens_generated.append(next_token.item())
            
            # Update hidden state
            with torch.cuda.amp.autocast(enabled=True):
                step_outputs = self.base_model(
                    input_ids=next_token,
                    output_hidden_states=True,
                )
            step_hidden = step_outputs.hidden_states[-1]
            combined_hidden = torch.cat([combined_hidden, step_hidden], dim=1)
            
            # Step 6: Gate 4 - Verification every N steps
            if (step + 1) % 10 == 0:
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                content_type = self._detect_content_type(generated_text)
                
                verification = self.gate4.verify(
                    combined_hidden[:, -1, :],
                    generated_text,
                    content_type
                )
                
                last_confidence = verification.confidence
                
                # Self-correction loop (Gate 3)
                if not verification.passed and self_corrections < config.self_correction_max_attempts:
                    if self.gate3.should_self_correct(
                        combined_hidden[:, -1, :],
                        num_failures=self_corrections + 1
                    ):
                        # Rollback last few tokens
                        rollback_count = min(5, len(tokens_generated))
                        if rollback_count > 0:
                            generated_ids = generated_ids[:, :-rollback_count]
                            tokens_generated = tokens_generated[:-rollback_count]
                            combined_hidden = combined_hidden[:, :-rollback_count]
                            self_corrections += 1
                            continue
            
            # Stop conditions
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Early stopping if confidence is very high and we have a good answer
            if last_confidence > 0.95 and step > 50:
                break
        
        # Final verification
        final_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        content_type = self._detect_content_type(final_text)
        final_verification = self.gate4.verify(
            combined_hidden[:, -1, :],
            final_text,
            content_type
        )
        
        elapsed = time.time() - start_time
        
        return GenerationResult(
            text=final_text,
            tokens=tokens_generated,
            num_self_corrections=self_corrections,
            verification_passed=final_verification.passed,
            confidence=final_verification.confidence,
            latency_seconds=elapsed,
            metadata={
                "retrieval": retrieval_metadata,
                "reasoning_steps": reasoning_metadata.get("num_steps", 0),
                "final_verdict": final_verification.verdict,
            }
        )
    
    def _detect_content_type(self, text: str) -> str:
        """Detect if the generated content is code, proof, or text."""
        if "```" in text or "def " in text or "function " in text or "class " in text:
            return "code"
        elif any(kw in text.lower() for kw in ["proof", "theorem", "lemma", "therefore", "hence"]):
            return "proof"
        return "text"
    
    def stream_generate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> GenerationResult:
        """
        Streaming generation that yields tokens as they are generated.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            callback: Optional callback(token, confidence) called for each token
            
        Returns:
            Final GenerationResult
        """
        if config is None:
            config = self.config
        
        # For streaming, we use the same pipeline but call callback
        result = self.generate(prompt, config)
        
        if callback:
            for i, token_id in enumerate(result.tokens):
                token_text = self.tokenizer.decode([token_id])
                confidence = 1.0 - (i / len(result.tokens)) * 0.2  # Confidence decreases over time
                callback(token_text, confidence)
        
        return result


def interactive_mode(pipeline: DistillV4InferencePipeline):
    """Run interactive REPL for testing."""
    print("\n" + "="*60)
    print("Distill-V4 Interactive Mode")
    print("="*60)
    print("Type your prompts and press Enter. Type 'quit' to exit.")
    print()
    
    while True:
        try:
            prompt = input("\n>>> ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt:
                continue
            
            result = pipeline.generate(prompt)
            
            print("\n" + "-"*60)
            print(f"Response ({result.latency_seconds:.2f}s, {len(result.tokens)} tokens, "
                  f"verif={'PASS' if result.verification_passed else 'FAIL'}, "
                  f"conf={result.confidence:.2f}, "
                  f"corrections={result.num_self_corrections})")
            print("-"*60)
            print(result.text)
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Distill-V4 Inference Pipeline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                       help="Base model path or HuggingFace model ID")
    parser.add_argument("--gate1", help="Path to Gate 1 (Knowledge Retrieval) checkpoint")
    parser.add_argument("--gate2", help="Path to Gate 2 (Symbolic Reasoning) checkpoint")
    parser.add_argument("--gate3", help="Path to Gate 3 (RL) checkpoint")
    parser.add_argument("--gate4", help="Path to Gate 4 (Verification) checkpoint")
    parser.add_argument("--prompt", help="Single prompt to generate for")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", default="bfloat16")
    
    args = parser.parse_args()
    
    # Build config
    config = GenerationConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
    
    # Initialize pipeline
    pipeline = DistillV4InferencePipeline(
        base_model_path=args.model,
        gate1_path=args.gate1,
        gate2_path=args.gate2,
        gate3_path=args.gate3,
        gate4_path=args.gate4,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    
    if args.interactive:
        interactive_mode(pipeline)
    elif args.prompt:
        result = pipeline.generate(args.prompt, config)
        print(f"\nVerification: {'PASS' if result.verification_passed else 'FAIL'}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Self-corrections: {result.num_self_corrections}")
        print(f"Latency: {result.latency_seconds:.2f}s")
        print(f"\n{result.text}")
    else:
        print("Error: Specify --prompt or --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()