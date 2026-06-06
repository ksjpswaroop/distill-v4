#!/usr/bin/env python3
"""
Phase 1: Collect Distillation Data from DeepSeek-V4

This script collects (question, response) pairs from DeepSeek-V4 API
for knowledge distillation. Filters for English-only, coding/reasoning content.

Usage:
    python scripts/collect_distillation_data.py --phase=sft --output data/sft/train.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import anthropic
import yaml


@dataclass
class Problem:
    """A distillation problem with metadata."""
    problem: str
    category: str
    difficulty: str
    
    def to_prompt(self) -> str:
        return f"""Problem ({self.category}, {self.difficulty}):
{self.problem}

Provide a detailed solution with code and reasoning."""


# Problem banks by category
PROBLEM_BANKS = {
    "code_generation": [
        Problem(
            problem="Implement a function to find the longest palindromic substring in a string.",
            category="code_generation",
            difficulty="medium"
        ),
        Problem(
            problem="Write a function that merges two sorted linked lists into one sorted linked list.",
            category="code_generation",
            difficulty="easy"
        ),
        Problem(
            problem="Implement a thread-safe LRU cache with O(1) get and put operations.",
            category="code_generation",
            difficulty="hard"
        ),
        Problem(
            problem="Write a function to serialize and deserialize a binary tree.",
            category="code_generation",
            difficulty="medium"
        ),
        Problem(
            problem="Implement a regex matcher supporting '.' and '*' wildcard characters.",
            category="code_generation",
            difficulty="medium"
        ),
        Problem(
            problem="Write a function to find the kth largest element in an unsorted array.",
            category="code_generation",
            difficulty="easy"
        ),
        Problem(
            problem="Implement a trie data structure with insert, search, and startsWith methods.",
            category="code_generation",
            difficulty="medium"
        ),
        Problem(
            problem="Write a function to detect a cycle in a directed graph.",
            category="code_generation",
            difficulty="medium"
        ),
    ],
    "algorithm_problems": [
        Problem(
            problem="Given an array of integers, find all triplets that sum to zero. Return distinct triplets.",
            category="algorithm_problems",
            difficulty="medium"
        ),
        Problem(
            problem="Find the minimum number of coins needed to make a given amount using unlimited coins of denominations.",
            category="algorithm_problems",
            difficulty="medium"
        ),
        Problem(
            problem="Given two strings, find the length of the longest common subsequence.",
            category="algorithm_problems",
            difficulty="medium"
        ),
        Problem(
            problem="Find the median of two sorted arrays in O(log(min(m,n))) time.",
            category="algorithm_problems",
            difficulty="hard"
        ),
        Problem(
            problem="Given a 2D matrix with sorted rows and columns, search for a target value.",
            category="algorithm_problems",
            difficulty="medium"
        ),
        Problem(
            problem="Find all permutations of a string with distinct characters.",
            category="algorithm_problems",
            difficulty="medium"
        ),
        Problem(
            problem="Solve the N-Queens problem and return all distinct solutions.",
            category="algorithm_problems",
            difficulty="hard"
        ),
    ],
    "formal_proofs": [
        Problem(
            problem="Prove that quicksort has O(n log n) average time complexity.",
            category="formal_proofs",
            difficulty="hard"
        ),
        Problem(
            problem="Prove that a binary search tree has O(log n) average search time.",
            category="formal_proofs",
            difficulty="medium"
        ),
        Problem(
            problem="Prove that the sum of the first n natural numbers is n(n+1)/2 using mathematical induction.",
            category="formal_proofs",
            difficulty="easy"
        ),
        Problem(
            problem="Prove that if a graph is bipartite, it contains no odd-length cycles.",
            category="formal_proofs",
            difficulty="medium"
        ),
        Problem(
            problem="Prove that any connected graph with n vertices has at least n-1 edges.",
            category="formal_proofs",
            difficulty="medium"
        ),
        Problem(
            problem="Prove that the runtime of merge sort is O(n log n) in all cases.",
            category="formal_proofs",
            difficulty="medium"
        ),
        Problem(
            problem="Use FOL to formalize and prove: All humans are mortal, Socrates is human, therefore Socrates is mortal.",
            category="formal_proofs",
            difficulty="medium"
        ),
    ],
    "math_reasoning": [
        Problem(
            problem="Find the number of ways to climb n stairs if you can take 1, 2, or 3 steps at a time.",
            category="math_reasoning",
            difficulty="medium"
        ),
        Problem(
            problem="Calculate the probability of getting exactly k heads in n coin flips.",
            category="math_reasoning",
            difficulty="easy"
        ),
        Problem(
            problem="Find the number of distinct ways to partition a set of n elements.",
            category="math_reasoning",
            difficulty="hard"
        ),
        Problem(
            problem="Prove that the square root of 2 is irrational using proof by contradiction.",
            category="math_reasoning",
            difficulty="medium"
        ),
        Problem(
            problem="Solve: Find the closed form of the recurrence T(n) = 2T(n/2) + n.",
            category="math_reasoning",
            difficulty="medium"
        ),
        Problem(
            problem="Find the number of prime numbers less than n using the Sieve of Eratosthenes.",
            category="math_reasoning",
            difficulty="easy"
        ),
        Problem(
            problem="Prove that there are infinitely many prime numbers using Euclid's argument.",
            category="math_reasoning",
            difficulty="medium"
        ),
    ],
    "debugging": [
        Problem(
            problem="Debug this Python function that finds the index of a target in a sorted array. It returns -1 but should return the correct index: def binary_search(arr, target): left, right = 0, len(arr)-1 while left <= right: mid = (left + right) // 2 if arr[mid] == target: return mid elif arr[mid] < target: left = mid + 1 else: right = mid return -1",
            category="debugging",
            difficulty="easy"
        ),
        Problem(
            problem="Debug this function that checks if a string has all unique characters. It sometimes returns wrong results: def is_unique(s): chars = set() for c in s: if c in chars: return False chars.add(c) return True # Try is_unique('abA')",
            category="debugging",
            difficulty="easy"
        ),
        Problem(
            problem="Debug this function that reverses a linked list. It creates an infinite loop: def reverse(head): prev = None current = head while current: next_node = current.next current.next = prev prev = current current = next_node return prev",
            category="debugging",
            difficulty="medium"
        ),
        Problem(
            problem="Debug this code that checks if a parentheses string is balanced. It fails for ')()(': def is_balanced(s): stack = [] for c in s: if c == '(': stack.append(c) elif c == ')': if not stack: return False stack.pop() return len(stack) == 0",
            category="debugging",
            difficulty="medium"
        ),
    ],
    "refinement": [
        Problem(
            problem="Optimize this function that checks if two strings are anagrams. Make it O(n) instead of O(n log n): def are_anagrams(s1, s2): return sorted(s1) == sorted(s2)",
            category="refinement",
            difficulty="easy"
        ),
        Problem(
            problem="Refactor this nested loop to improve time complexity: def find_duplicates(nums): result = [] for i in range(len(nums)): for j in range(i+1, len(nums)): if nums[i] == nums[j]: result.append(nums[i]) return result",
            category="refinement",
            difficulty="medium"
        ),
        Problem(
            problem="Make this recursive Fibonacci implementation efficient using memoization: def fib(n): if n <= 1: return n return fib(n-1) + fib(n-2)",
            category="refinement",
            difficulty="easy"
        ),
        Problem(
            problem="Reduce space complexity of this function from O(n) to O(1): def find_max_subarray_sum(arr, k): max_sum = 0 window_sum = sum(arr[:k]) for i in range(k, len(arr)): window_sum += arr[i] - arr[i-k] max_sum = max(max_sum, window_sum) return max_sum",
            category="refinement",
            difficulty="medium"
        ),
    ],
}

# Scale up problem banks programmatically
def generate_problem_variants() -> list[Problem]:
    """Generate additional problem variants to reach target dataset size."""
    variants = []
    
    # Code generation patterns
    patterns = [
        ("Implement {algo} in {lang} with O({complexity}) complexity", "code_generation"),
        ("Write a {lang} function to {task}", "code_generation"),
        ("Create a {data_structure} with {ops} operations", "code_generation"),
        ("Implement {alg} algorithm for {problem_type}", "code_generation"),
    ]
    
    for pattern, cat in patterns:
        for i in range(50):
            variants.append(Problem(
                problem=f"{pattern} (variant {i+1})",
                category=cat,
                difficulty=random.choice(["easy", "medium", "hard"])
            ))
    
    # Algorithm complexity proofs
    for i in range(30):
        variants.append(Problem(
            problem=f"Prove that {random.choice(['merge sort', 'quicksort', 'heapsort', 'binary search'])} has {random.choice(['O(n log n)', 'O(n)', 'O(log n)'])} complexity in the {random.choice(['best', 'average', 'worst'])} case",
            category="formal_proofs",
            difficulty=random.choice(["medium", "hard"])
        ))
    
    # Mathematical reasoning
    for i in range(40):
        variants.append(Problem(
            problem=f"Solve: {random.choice(['Find the number of ways to', 'Calculate the probability of', 'Determine the closed form of'])} {random.choice(['arranging', 'selecting', 'partitioning'])} {random.choice(['n', '2n', 'n+1'])} items with {random.choice(['specific constraints', 'no restrictions', 'the following properties'])}",
            category="math_reasoning",
            difficulty=random.choice(["easy", "medium", "hard"])
        ))
    
    return variants


class DistillationCollector:
    """Collects distillation data from DeepSeek-V4 API."""
    
    def __init__(self, api_key: Optional[str] = None, config_path: str = "configs/config.yaml"):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        
        if Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.stats = {"collected": 0, "failed": 0, "filtered": 0}
    
    def collect_sample(self, problem: Problem, model: str = "claude-sonnet-4-20250514") -> Optional[dict]:
        """Collect a single sample from the API."""
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=8192,
                temperature=0.7,
                system="""You are an expert coding and reasoning assistant. 
You specialize in:
- Writing clean, efficient, well-documented code
- Mathematical proofs and formal reasoning
- Algorithm analysis and complexity computation
- Debugging and code optimization

When providing code:
1. Include the implementation
2. Add inline comments explaining key steps
3. Include test cases
4. Explain the time and space complexity

When providing proofs:
1. State the theorem clearly
2. Provide step-by-step reasoning
3. Conclude with the proof result

Always be precise and thorough.""",
                messages=[
                    {"role": "user", "content": problem.to_prompt()}
                ]
            )
            
            return {
                "problem": problem.problem,
                "category": problem.category,
                "difficulty": problem.difficulty,
                "response": response.content[0].text,
                "model_used": model,
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
            
        except Exception as e:
            print(f"Error collecting sample: {e}", file=sys.stderr)
            self.stats["failed"] += 1
            return None
    
    def is_english(self, text: str) -> bool:
        """Basic English language detection."""
        # Simple heuristic: check for common English words
        english_indicators = ["the", "is", "are", "and", "or", "to", "of", "in", "that", "it"]
        text_lower = text.lower()
        count = sum(1 for word in english_indicators if word in text_lower)
        return count >= 3
    
    def is_quality_response(self, response: str) -> bool:
        """Filter for quality responses."""
        # Must have code blocks
        if "```" not in response:
            return False
        
        # Must have substantial content
        if len(response.split()) < 100:
            return False
        
        # Must be in English
        if not self.is_english(response):
            return False
        
        # Must have reasoning
        reasoning_indicators = ["because", "therefore", "thus", "hence", "proof", "reasoning"]
        if not any(word in response.lower() for word in reasoning_indicators):
            return False
        
        return True
    
    def collect_dataset(
        self, 
        output_path: str, 
        target_size: int = 50000,
        rate_limit_per_minute: int = 60
    ) -> None:
        """Collect a full distillation dataset."""
        
        all_problems = []
        
        # Add bank problems
        for category, problems in PROBLEM_BANKS.items():
            all_problems.extend(problems)
        
        # Add generated variants
        all_problems.extend(generate_problem_variants())
        
        # Replicate to reach target size
        while len(all_problems) < target_size:
            all_problems.extend(all_problems)
        
        all_problems = all_problems[:target_size]
        random.shuffle(all_problems)
        
        print(f"Collecting {len(all_problems)} problems to {output_path}")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        rate_limiter = RateLimiter(rate_limit_per_minute)
        
        with open(output_file, "w") as f:
            for i, problem in enumerate(all_problems):
                if i % 100 == 0:
                    print(f"Progress: {i}/{len(all_problems)} "
                          f"(collected={self.stats['collected']}, "
                          f"failed={self.stats['failed']}, "
                          f"filtered={self.stats['filtered']})")
                
                rate_limiter.wait_if_needed()
                
                sample = self.collect_sample(problem)
                
                if sample is None:
                    continue
                
                if not self.is_quality_response(sample["response"]):
                    self.stats["filtered"] += 1
                    continue
                
                f.write(json.dumps(sample) + "\n")
                f.flush()  # Ensure we don't lose data
                self.stats["collected"] += 1
                
                # Small delay between requests
                time.sleep(0.5)
        
        print(f"\nCollection complete:")
        print(f"  Collected: {self.stats['collected']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Filtered: {self.stats['filtered']}")
        print(f"  Output: {output_file}")


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.interval = 60.0 / max_per_minute
        self.last_call = 0.0
    
    def wait_if_needed(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


def main():
    parser = argparse.ArgumentParser(description="Collect distillation data from DeepSeek-V4")
    parser.add_argument("--phase", choices=["sft", "rl"], default="sft",
                       help="Distillation phase (affects problem types)")
    parser.add_argument("--output", default="data/sft/train.jsonl",
                       help="Output file path")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Config file path")
    parser.add_argument("--target-size", type=int, default=50000,
                       help="Target number of samples")
    parser.add_argument("--rate-limit", type=int, default=60,
                       help="API rate limit per minute")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                       help="Model to use for collection")
    
    args = parser.parse_args()
    
    collector = DistillationCollector(config_path=args.config)
    collector.collect_dataset(
        output_path=args.output,
        target_size=args.target_size,
        rate_limit_per_minute=args.rate_limit
    )


if __name__ == "__main__":
    main()