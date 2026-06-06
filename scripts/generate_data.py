"""
Data generation pipeline for Distill-V4 gate training.

Generates synthetic (question, DeepSeek-response) pairs for:
  - Gate 1: Retrieval training (fact lookup, code patterns)
  - Gate 2: FOL reasoning (logic, math, proofs)
  - Gate 3: RL reward signals (correctness, efficiency)
  - Gate 4: Verification (correct + incorrect examples)

Supports:
  - DeepSeek V3 API
  - OpenAI API (o3-mini, GPT-4o)
  - Local generation via LiteLLM
"""

import json
import os
import sys
import hashlib
import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import torch
import numpy as np

# ---- Data categories and templates ----

CODING_CATEGORIES = [
    "implementation", "debugging", "optimization", "algorithm",
    "data_structure", "system_design", "code_review", "testing"
]

MATH_CATEGORIES = [
    "algebra", "calculus", "linear_algebra", "combinatorics",
    "number_theory", "probability", "geometry", "proofs"
]

LOGIC_CATEGORIES = [
    "propositional", "first_order_logic", "syllogism",
    "abduction", "induction", "deduction", "common_sense"
]

REASONING_CATEGORIES = [
    "chain_of_thought", "analogical", "causal", "spatial",
    "temporal", "commonsense", "planning", "decision"
]

ALL_CATEGORIES = {
    "coding": CODING_CATEGORIES,
    "math": MATH_CATEGORIES,
    "logic": LOGIC_CATEGORIES,
    "reasoning": REASONING_CATEGORIES,
}

PROBLEM_TEMPLATES = {
    "coding": [
        "Write a Python function to {task} with O({complexity}) time complexity.",
        "Implement a {data_structure} class in Python with methods: {methods}.",
        "Debug the following Python code: {code}. Find and fix the bug.",
        "Optimize this {algorithm} implementation for {constraint}: {code}",
        "Write unit tests for: {function_signature}",
        "Explain what this code does: {code}",
    ],
    "math": [
        "Solve for x: {equation}",
        "Prove that {statement} using {method}.",
        "Find the {quantity} of {object} given that {conditions}.",
        "Evaluate {expression} where {conditions}.",
        "How many ways can {event} occur if {conditions}?",
    ],
    "logic": [
        "Given premises: {premises}. What conclusion follows?",
        "Is this argument valid? {argument}. Explain your reasoning.",
        "Formalize this statement in FOL: {statement}",
        "Determine if {statement1} entails {statement2}. Justify.",
        "Find the error in this proof: {proof}",
    ],
    "reasoning": [
        "If {premise1} and {premise2}, then what is the most likely {conclusion_type}?",
        "What is the next step in this plan? {plan}",
        "Given these observations: {observations}. What explains {phenomenon}?",
        "Which option is most consistent with {constraints}? {options}",
        "Complete the pattern: {sequence}",
    ],
}


@dataclass
class DistillationSample:
    """A single distillation training example."""
    question: str
    reference_answer: str
    category: str
    subcategory: str
    language: str
    difficulty: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["_hash"] = self.hash()
        return d

    def hash(self) -> str:
        return hashlib.sha256(
            f"{self.question}|{self.reference_answer}".encode()
        ).hexdigest()[:16]


class DataGenerator:
    """
    Generates distillation data using LLM APIs.

    Falls back to template-based generation if API is unavailable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        rate_limit: int = 10,  # requests per second
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base
        self.model = model
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.last_request_time = 0.0
        self._executor = ThreadPoolExecutor(max_workers=rate_limit)

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def generate_sample(
        self,
        category: str,
        subcategory: str,
        difficulty: str = "medium",
    ) -> Optional[DistillationSample]:
        """Generate a single sample using API or templates."""

        # Try API first
        if self.api_key:
            try:
                return self._generate_via_api(category, subcategory, difficulty)
            except Exception as e:
                print(f"API error: {e}, falling back to template", file=sys.stderr)

        # Fall back to template-based generation
        return self._generate_via_template(category, subcategory, difficulty)

    def _generate_via_api(
        self,
        category: str,
        subcategory: str,
        difficulty: str,
    ) -> DistillationSample:
        """Generate via DeepSeek/OpenAI API."""
        self._rate_limit()

        prompt = self._build_prompt(category, subcategory, difficulty)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert coding and reasoning assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        import requests
        response = requests.post(
            f"{self.api_base}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return DistillationSample(
            question=prompt,
            reference_answer=answer,
            category=category,
            subcategory=subcategory,
            language="en",
            difficulty=difficulty,
            metadata={"source": "api", "model": self.model},
        )

    def _build_prompt(
        self,
        category: str,
        subcategory: str,
        difficulty: str,
    ) -> str:
        """Build a problem prompt for the given category."""
        templates = PROBLEM_TEMPLATES.get(category, PROBLEM_TEMPLATES["reasoning"])
        template = random.choice(templates)

        # Fill in template slots
        difficulty_modifier = {
            "easy": "simple, few constraints",
            "medium": "moderate complexity",
            "hard": "challenging, edge cases required",
        }.get(difficulty, "moderate complexity")

        fillers = {
            "task": random.choice([
                "reverse a linked list", "compute all permutations",
                "find the longest palindromic substring", "implement a LRU cache",
                "flatten a nested list", "serialize a binary tree",
            ]),
            "complexity": random.choice(["n", "n log n", "n^2", "1"]),
            "data_structure": random.choice(["Binary Tree", "Graph", "Hash Map", "Trie"]),
            "methods": "insert, delete, search, traverse",
            "code": "# TODO: implement quicksort\ndef quicksort(arr):\n    pass",
            "function_signature": "def fibonacci(n: int) -> int:",
            "algorithm": random.choice(["dynamic programming", "greedy", "divide and conquer"]),
            "constraint": "memory efficiency",
            "equation": "2x^2 - 5x + 2 = 0",
            "statement": "the sum of two even numbers is even",
            "method": random.choice(["direct proof", "induction", "contraposition"]),
            "quantity": "determinant",
            "object": "matrix A",
            "conditions": "A is a 2x2 matrix with entries [1,2;3,4]",
            "expression": "\\int_0^1 x^2 dx",
            "event": "arranging the word ALGORITHM",
            "premises": "All humans are mortal. Socrates is human.",
            "argument": "If it rains, the ground is wet. The ground is wet. Therefore it rained.",
            "statement1": "\\forall x (P(x) \\to Q(x))",
            "statement2": "\\exists x (P(x) \\land Q(x))",
            "proof": "Proof by induction that all horses are the same color...",
            "premise1": "It is raining",
            "premise2": "The ground is wet",
            "conclusion_type": "explanation",
            "plan": "First heat the pan, then add oil, finally cook the eggs.",
            "observations": "The sky is cloudy, the barometric pressure is dropping",
            "phenomenon": "the weather change",
            "constraints": "limited budget and time",
            "options": "A) Postpone B) Rush C) Cancel D) Proceed",
            "sequence": "2, 4, 8, 16, ?",
        }

        prompt = template
        for key, value in fillers.items():
            prompt = prompt.replace(f"{{{key}}}", value)

        return prompt

    def _generate_via_template(
        self,
        category: str,
        subcategory: str,
        difficulty: str,
    ) -> DistillationSample:
        """Generate via template only (no API)."""
        question = self._build_prompt(category, subcategory, difficulty)
        # Generate a plausible (but not API-sourced) answer
        answer = f"[Template-generated answer for {category}/{subcategory}]\n\nThis is a placeholder answer generated by template. Use API for real responses."

        return DistillationSample(
            question=question,
            reference_answer=answer,
            category=category,
            subcategory=subcategory,
            language="en",
            difficulty=difficulty,
            metadata={"source": "template"},
        )


def generate_dataset(
    num_samples: int,
    output_path: str,
    category_weights: Optional[Dict[str, float]] = None,
    difficulty_weights: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
    parallel: int = 8,
):
    """
    Generate a full distillation dataset.

    Args:
        num_samples: Total number of samples
        output_path: Where to save the JSONL file
        category_weights: Sampling weights for categories
        difficulty_weights: Sampling weights for difficulties
        api_key: API key for LLM generation
        parallel: Number of parallel API workers
    """
    category_weights = category_weights or {
        "coding": 0.50,
        "math": 0.25,
        "logic": 0.10,
        "reasoning": 0.15,
    }
    difficulty_weights = difficulty_weights or {
        "easy": 0.20,
        "medium": 0.50,
        "hard": 0.30,
    }

    generator = DataGenerator(api_key=api_key)

    categories = list(category_weights.keys())
    cat_probs = list(category_weights.values())

    difficulties = list(difficulty_weights.keys())
    diff_probs = list(difficulty_weights.values())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for i in range(num_samples):
            # Sample category and difficulty
            cat = random.choices(categories, weights=cat_probs)[0]
            subcats = ALL_CATEGORIES[cat]
            subcat = random.choice(subcats)
            diff = random.choices(difficulties, weights=diff_probs)[0]

            sample = generator.generate_sample(cat, subcat, diff)

            if sample:
                f.write(json.dumps(sample.to_dict()) + "\n")

            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{num_samples} samples", flush=True)

    print(f"Dataset saved to {output_path} ({num_samples} samples)")


def filter_english_only(input_path: str, output_path: str, min_ratio: float = 0.95):
    """
    Filter dataset to English-only content.
    Uses simple heuristics: checks for common non-English character ranges.
    """
    import re

    non_english_pattern = re.compile(r'[^\x00-\x7F]')
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    arabic_pattern = re.compile(r'[\u0600-\u06ff]')
    cyrillic_pattern = re.compile(r'[\u0400-\u04ff]')

    kept = 0
    removed = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            sample = json.loads(line)
            text = sample["question"] + " " + sample["reference_answer"]

            # Count non-ASCII characters
            non_ascii_chars = len(non_english_pattern.findall(text))
            total_chars = len(text)
            english_ratio = 1 - non_ascii_chars / max(total_chars, 1)

            if english_ratio >= min_ratio:
                fout.write(line)
                kept += 1
            else:
                removed += 1

    print(f"English filter: kept {kept}, removed {removed} ({kept/(kept+removed)*100:.1f}%)")


def deduplicate(input_path: str, output_path: str):
    """Remove duplicate questions by hash."""
    seen_hashes = set()
    unique = 0
    dupes = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            sample = json.loads(line)
            h = sample.get("_hash") or hashlib.sha256(
                sample["question"].encode()
            ).hexdigest()[:16]

            if h not in seen_hashes:
                seen_hashes.add(h)
                fout.write(line)
                unique += 1
            else:
                dupes += 1

    print(f"Dedup: {unique} unique, {dupes} duplicates removed")


def split_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.90,
    eval_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
):
    """Split dataset into train/eval/test."""
    import math

    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    with open(input_path, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)

    n = len(lines)
    n_train = int(n * train_ratio)
    n_eval = int(n * eval_ratio)

    splits = {
        "train": lines[:n_train],
        "eval": lines[n_train:n_train + n_eval],
        "test": lines[n_train + n_eval:],
    }

    os.makedirs(output_dir, exist_ok=True)
    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, 'w') as f:
            f.writelines(data)
        print(f"  {name}: {len(data)} samples -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill-V4 Data Generation")
    parser.add_argument("--mode", choices=["generate", "filter", "dedup", "split"], required=True)
    parser.add_argument("--input", type=str, help="Input file for filter/dedup/split")
    parser.add_argument("--output", type=str, help="Output file/dir")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "generate":
        generate_dataset(
            num_samples=args.num_samples,
            output_path=args.output,
            api_key=args.api_key,
            parallel=args.parallel,
        )
    elif args.mode == "filter":
        filter_english_only(args.input, args.output)
    elif args.mode == "dedup":
        deduplicate(args.input, args.output)
    elif args.mode == "split":
        split_dataset(
            input_path=args.input,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            eval_ratio=args.eval_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )