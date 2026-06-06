"""
Gate Module Definitions for Distill-V4

Each gate is a trainable module that transforms/augments the base model's hidden states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class KnowledgeRetrievalGate(nn.Module):
    """
    Gate 1: Knowledge Retrieval (2B params)
    
    Dynamic retrieval of relevant facts, code patterns, and problem-solving 
    strategies from episodic memory and external knowledge bases.
    
    Architecture:
    - Attention-based retrieval controller
    - Episodic memory bank (learned embeddings)
    - Knowledge fusion gate (learned merge)
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        memory_size: int = 100_000,
        num_heads: int = 16,
        key_dim: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        
        # Query projection from hidden state
        self.query_proj = nn.Linear(hidden_dim, key_dim * num_heads, bias=False)
        
        # Memory key/value projections
        self.key_proj = nn.Linear(hidden_dim, key_dim * num_heads, bias=False)
        self.value_proj = nn.Linear(hidden_dim, key_dim * num_heads, bias=False)
        
        # Learned episodic memory bank
        self.register_buffer(
            "memory_bank",
            torch.randn(memory_size, key_dim * num_heads) * 0.02
        )
        
        # Output projection
        self.out_proj = nn.Linear(key_dim * num_heads, hidden_dim, bias=False)
        
        # Knowledge fusion gate
        # g = sigmoid(W_fuse @ [h; retrieved])
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Relevance scoring head
        self.relevance_scorer = nn.Sequential(
            nn.Linear(key_dim, 1),
            nn.Sigmoid()
        )
        
        self.scale = key_dim ** -0.5
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        input_tokens: torch.Tensor,
        retrieve_top_k: int = 32
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden_state: (batch, seq_len, hidden) from base LM
            input_tokens: (batch, seq_len) raw tokens for retrieval context
            retrieve_top_k: number of memory entries to retrieve
            
        Returns:
            fused_hidden: hidden state enriched with retrieved knowledge
            metadata: retrieval statistics and retrieved entries
        """
        batch_size, seq_len, _ = hidden_state.shape
        
        # Use last token's hidden state as query
        query_input = hidden_state[:, -1, :]  # (batch, hidden)
        query = self.query_proj(query_input)  # (batch, key_dim * num_heads)
        query = query.view(batch_size, self.num_heads, self.key_dim)
        
        # Retrieve from episodic memory
        memory_keys = self.memory_bank  # (memory_size, key_dim * num_heads)
        memory_keys = memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
        memory_keys = memory_keys.view(batch_size, self.memory_size, self.num_heads, self.key_dim)
        memory_keys = memory_keys.permute(0, 2, 1, 3)  # (batch, heads, memory, key_dim)
        
        # Attention scores
        query_for_attn = query.permute(0, 2, 1)  # (batch, key_dim, heads)
        scores = torch.matmul(query_for_attn.unsqueeze(-2), memory_keys).squeeze(-2)  # (batch, heads, memory)
        scores = scores * self.scale
        
        # Top-k attention
        top_k = min(retrieve_top_k, self.memory_size)
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)  # (batch, heads, top_k)
        
        # Get top-k values from memory
        memory_values = self.memory_bank.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        memory_values = memory_values.gather(2, top_indices.unsqueeze(-1).expand(-1, -1, -1, self.key_dim))
        memory_values = memory_values.permute(0, 1, 3, 2)  # (batch, heads, key_dim, top_k)
        
        # Softmax attention weights
        attn_weights = F.softmax(top_scores, dim=-1)  # (batch, heads, top_k)
        attn_weights = attn_weights.unsqueeze(2)  # (batch, heads, 1, top_k)
        
        # Weighted sum of retrieved values
        retrieved = (attn_weights * memory_values).sum(-1)  # (batch, heads, key_dim)
        retrieved = retrieved.permute(0, 2, 1).contiguous()  # (batch, key_dim, heads)
        retrieved = retrieved.view(batch_size, -1)  # (batch, key_dim * heads)
        
        # Output projection
        retrieved_hidden = self.out_proj(retrieved)  # (batch, hidden)
        
        # Relevance score for the retrieval
        relevance = self.relevance_scorer(
            (query * retrieved_hidden[:, :self.key_dim]).sum(-1) / self.key_dim
        )
        
        # Fusion gate
        combined = torch.cat([hidden_state[:, -1, :], retrieved_hidden], dim=-1)
        gate = self.fusion_gate(combined)  # (batch, hidden)
        
        # Gated combination
        fused_hidden = gate * retrieved_hidden + (1 - gate) * hidden_state[:, -1, :]
        
        return fused_hidden, {
            "relevance": relevance,
            "top_indices": top_indices,
            "attention_scores": top_scores,
        }
    
    def update_memory(self, new_entries: torch.Tensor, importance_scores: torch.Tensor):
        """
        Update episodic memory with new entries (for online learning).
        
        Args:
            new_entries: (batch, hidden) new memory entries
            importance_scores: (batch,) importance of each entry
        """
        # Find lowest-importance entries to replace
        with torch.no_grad():
            values, indices = torch.topk(importance_scores, k=min(100, len(importance_scores)), largest=False)
            for idx, (entry, score) in enumerate(zip(new_entries, importance_scores)):
                replace_idx = indices[idx % len(indices)].item()
                self.memory_bank[replace_idx] = entry.detach()
                self.memory_weights[replace_idx] = score.detach()


class SymbolicReasoningGate(nn.Module):
    """
    Gate 2: Symbolic Reasoning with First-Order Logic (4B params)
    
    Formal FOL reasoning, natural logic inference, and proof generation.
    
    Sub-modules:
    - FOL Formalizer (1B): NL → FOL conversion
    - Natural Logic Inferencer (1B): Entailment, contradiction detection
    - Symbolic Reasoner (1.5B): Theorem proving, rewriting
    - Proof Validator (0.5B): Proof correctness checking
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        intermediate_dim: int = 16384,
        num_reasoning_steps: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasoning_steps = num_reasoning_steps
        
        # FOL Formalizer: Converts natural language to FOL
        self.fol_formalizer = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )
        
        # Natural Logic Inferencer: Entailment/contradiction detection
        self.natlog_entailment = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # entail, contradict, neutral
        )
        
        # Symbolic Reasoner: Neural theorem prover interface
        self.symbolic_prover = NeuralTheoremProver(hidden_dim, intermediate_dim)
        
        # Proof Validator: Checks proof correctness
        self.proof_validator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Reasoning chain aggregator
        self.chain_aggregator = nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
        )
        
        # Step projection
        self.step_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        hidden_state: torch.Tensor,
        reasoning_context: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden_state: (batch, seq, hidden) from base LM
            reasoning_context: Natural language reasoning context
            
        Returns:
            reasoning_output: FOL-enriched hidden state
            metadata: proof trace, validity scores
        """
        batch_size = hidden_state.size(0)
        
        # Formalize reasoning as FOL
        fol_hidden = self.fol_formalizer(hidden_state[:, -1, :])
        
        # Build reasoning chain
        proof_steps = []
        current_state = fol_hidden
        chain_hidden = torch.zeros_like(current_state)
        
        for step in range(self.num_reasoning_steps):
            # Generate next reasoning step
            step_input = current_state + chain_hidden
            step_hidden = self.step_proj(step_input)
            
            # Symbolic reasoning step
            step_result = self.symbolic_prover.step(step_hidden, proof_steps)
            proof_steps.append(step_result)
            
            # Update chain state
            chain_hidden = self.chain_aggregator(step_result, chain_hidden)
        
        # Validate the complete proof
        final_proof_state = torch.stack([s["hidden"] for s in proof_steps], dim=1).mean(dim=1)
        proof_validity = self.proof_validator(final_proof_state)
        
        # Check logical consistency between steps
        entailment_scores = []
        for i in range(len(proof_steps) - 1):
            combined = torch.cat([proof_steps[i]["hidden"], proof_steps[i+1]["hidden"]], dim=-1)
            entailment = self.natlog_entailment(combined)
            entailment_scores.append(entailment)
        
        return chain_hidden, {
            "proof_steps": proof_steps,
            "proof_validity": proof_validity,
            "entailment_scores": torch.stack(entailment_scores) if entailment_scores else None,
            "num_steps": len(proof_steps),
        }


class NeuralTheoremProver(nn.Module):
    """Neural interface to symbolic theorem proving."""
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        
        # Resolution step generator
        self.resolution_step = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )
        
        # Unification predictor
        self.unification_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Rewrite rule selector
        self.rewrite_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),  # 32 rewrite rules
        )
    
    def step(
        self,
        current_state: torch.Tensor,
        proof_history: list,
    ) -> Dict[str, torch.Tensor]:
        """Generate one step of a proof."""
        
        # Generate resolution candidate
        resolution = self.resolution_step(current_state)
        
        # Predict unification score with previous steps
        unification_scores = []
        for prev_step in proof_history[-3:]:  # Look back 3 steps
            combined = torch.cat([current_state, prev_step["hidden"]], dim=-1)
            score = self.unification_predictor(combined)
            unification_scores.append(score)
        
        # Select rewrite rule
            rewrite_logits = self.rewrite_selector(current_state)
        rewrite_probs = F.softmax(rewrite_logits, dim=-1)
        
        return {
            "hidden": resolution,
            "rewrite_probs": rewrite_probs,
            "unification_scores": unification_scores,
        }


class RLGate(nn.Module):
    """
    Gate 3: Reinforcement Learning (1B params)
    
    PPO/GRPO-based reward shaping and self-correction.
    
    Components:
    - Reward estimator (learned reward from hidden state)
    - Value baseline
    - Self-correction predictor
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_votes: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_votes = num_votes
        
        # Reward estimator
        self.reward_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Value baseline for advantage computation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Self-correction predictor
        # Takes (current_hidden, error_hidden) → correction_needed
        self.correction_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # PPO clipping
        self.clip_epsilon = 0.2
    
    def compute_rewards(
        self,
        hidden_states: torch.Tensor,
        execution_results: list,
    ) -> torch.Tensor:
        """
        Compute reward signals from execution results.
        
        Args:
            hidden_states: (batch, seq, hidden) 
            execution_results: List of ExecutionResult objects
            
        Returns:
            rewards: (batch,) reward for each sample
        """
        # Use final hidden state
        final_hidden = hidden_states[:, -1, :]
        
        # Learned reward component
        learned_reward = self.reward_estimator(final_hidden).squeeze(-1)
        
        # Hardcoded reward components
        hardcoded_rewards = []
        for result in execution_results:
            # Execution correctness
            r_code = 1.0 if result.passed else -0.5
            # Test pass rate
            r_tests = result.test_pass_rate if hasattr(result, 'test_pass_rate') else 0.0
            # Efficiency bonus
            r_efficiency = 0.1 if result.runtime < result.expected_runtime else 0.0
            
            hardcoded = 0.4 * r_code + 0.4 * r_tests + 0.2 * r_efficiency
            hardcoded_rewards.append(hardcoded)
        
        hardcoded_rewards = torch.tensor(hardcoded_rewards, device=hidden_states.device)
        
        # Blend learned and hardcoded rewards
        rewards = 0.3 * learned_reward + 0.7 * hardcoded_rewards
        
        return rewards
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PPO advantages.
        
        Returns:
            advantages: (batch,)
            returns: (batch,) for policy update
        """
        # Generalized Advantage Estimation (GAE)
        advantages = []
        last_gae = 0
        
        # Simple advantage: reward - value baseline
        for i in range(len(rewards)):
            advantage = rewards[i] - values[i]
            advantages.append(advantage)
        
        advantages = torch.stack(advantages)
        returns = advantages + values  # For value function training
        
        return advantages, returns
    
    def should_self_correct(
        self,
        hidden_state: torch.Tensor,
        error_context: Optional[torch.Tensor] = None,
        num_failures: int = 0,
    ) -> bool:
        """
        Predict if self-correction should be attempted.
        
        Args:
            hidden_state: (batch, hidden)
            error_context: Optional error embedding
            num_failures: Number of consecutive verification failures
            
        Returns:
            correction_needed: bool
        """
        # If we've failed multiple times, don't keep correcting
        if num_failures >= 3:
            return False
        
        # Always correct after first failure
        if num_failures == 1:
            return True
        
        # Learn from hidden state
        if error_context is None:
            error_context = hidden_state  # Use current state as proxy
        
        combined = torch.cat([hidden_state, error_context], dim=-1)
        pred = self.correction_predictor(combined)
        
        return pred.item() > 0.5
    
    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute PPO clipped surrogate loss.
        
        Args:
            log_probs: (batch, seq) new action log probs
            old_log_probs: (batch, seq) old action log probs
            advantages: (batch,) advantage estimates
            mask: (batch, seq) optional mask for valid tokens
        """
        # Ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped objective
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.unsqueeze(-1)
        
        # Per-token loss, masked
        loss = -torch.min(surr1, surr2)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss


class VerificationGate(nn.Module):
    """
    Gate 4: Verification (3B params)
    
    Pre-token-streaming verification of generated content:
    - Code execution (sandboxed)
    - Proof checking (FOL)
    - Consistency validation (self-consistency voting)
    
    Sub-modules:
    - Code Executor (1B)
    - Proof Checker (1B)  
    - Consistency Checker (1B)
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_votes: int = 5,
        max_execution_time: float = 10.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_votes = num_votes
        self.max_execution_time = max_execution_time
        
        # Code Executor
        self.code_executor = None  # Initialized lazily for sandboxing
        
        # Test generator
        self.test_generator = TestCaseGenerator(hidden_dim)
        
        # Proof checker
        self.proof_checker = FormalProofChecker()
        
        # Consistency checker (self-consistency voting)
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Code correctness predictor
        self.code_correctness = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Proof validity predictor
        self.proof_validity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Verdict combiner (3 binary checks → final score)
        self.verdict_combiner = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def verify(
        self,
        hidden_state: torch.Tensor,
        generated_content: str,
        content_type: str,
    ) -> "VerificationResult":
        """
        Verify generated content before streaming.
        
        Args:
            hidden_state: (batch, hidden) current model state
            generated_content: The text/code to verify
            content_type: "code", "proof", or "text"
            
        Returns:
            VerificationResult with passed, verdict, feedback
        """
        results = {}
        
        if content_type == "code":
            # Neural code correctness prediction
            code_score = self.code_correctness(hidden_state)
            results["code_score"] = code_score.item()
            
            # TODO: Actual sandboxed execution when self.code_executor is set
            # For now, use neural prediction as proxy
            
        elif content_type == "proof":
            # Neural proof validity prediction
            proof_score = self.proof_validity(hidden_state)
            results["proof_score"] = proof_score.item()
        
        # Self-consistency check (always run)
        consistency_score = self.consistency_head(hidden_state)
        results["consistency_score"] = consistency_score.item()
        
        # Confidence
        confidence = self.confidence_estimator(hidden_state)
        results["confidence"] = confidence.item()
        
        # Combine verdicts
        code_passed = results.get("code_score", 1.0) > 0.5
        proof_passed = results.get("proof_score", 1.0) > 0.5
        consistent = results["consistency_score"] > 0.5
        
        verdict_input = torch.tensor([code_passed, proof_passed, consistent], dtype=torch.float32)
        final_verdict = self.verdict_combiner(verdict_input.unsqueeze(0)).item()
        
        passed = final_verdict > 0.5
        
        # Generate feedback
        feedback = None
        if not passed:
            failures = []
            if not code_passed:
                failures.append("code correctness check failed")
            if not proof_passed:
                failures.append("proof validity check failed")
            if not consistent:
                failures.append("self-consistency check failed")
            feedback = "; ".join(failures)
        
        return VerificationResult(
            passed=passed,
            verdict=final_verdict,
            confidence=results["confidence"],
            feedback=feedback,
            **results
        )
    
    def set_executor(self, executor):
        """Set the sandboxed code executor."""
        self.code_executor = executor


class VerificationResult:
    """Result from verification gate."""
    
    def __init__(
        self,
        passed: bool,
        verdict: float,
        confidence: float,
        feedback: Optional[str] = None,
        **kwargs
    ):
        self.passed = passed
        self.verdict = verdict
        self.confidence = confidence
        self.feedback = feedback
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"VerificationResult({status}, verdict={self.verdict:.2f})"


class TestCaseGenerator(nn.Module):
    """Generates test cases for code verification."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.test_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.test_count_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, code_hidden: torch.Tensor) -> list[str]:
        """Generate test cases from code hidden state."""
        test_hidden = self.test_generator(code_hidden)
        num_tests = max(1, int(self.test_count_head(test_hidden).sigmoid().item() * 5))
        
        # Placeholder - actual implementation would decode to test code
        return [f"test_case_{i}" for i in range(num_tests)]


class FormalProofChecker(nn.Module):
    """Neural FOL proof checker."""
    
    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        
        self.proof_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.axiom_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.inference_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, proof_hidden: torch.Tensor, axioms: list) -> dict:
        """Check a proof against axioms."""
        encoded = self.proof_encoder(proof_hidden)
        
        axiom_scores = []
        for axiom in axioms:
            score = self.axiom_head(torch.cat([encoded, axiom], dim=-1))
            axiom_scores.append(score)
        
        return {
            "axiom_scores": axiom_scores,
            "valid": all(s > 0.5 for s in axiom_scores)
        }


class ExecutionResult:
    """Result from code execution."""
    
    def __init__(
        self,
        passed: bool,
        output: str = "",
        error: Optional[str] = None,
        runtime: float = 0.0,
        test_pass_rate: float = 0.0,
    ):
        self.passed = passed
        self.output = output
        self.error = error
        self.runtime = runtime
        self.test_pass_rate = test_pass_rate