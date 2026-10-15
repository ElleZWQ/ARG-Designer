"""Single-agent baseline evaluation.

Measures the LLM's own ability on each benchmark WITHOUT training the
ARG-Designer graph generator. Each question is answered by ONE agent directly
(no topology, no multi-agent collaboration), matching the paper's
"Single-agent: Vanilla / CoT" baseline.

Usage:
    python experiment/baseline.py --domain gsm8k --llm_name GPT-5.6-Sol-joybuilder
    python experiment/baseline.py --domain aqua   --llm_name GPT-5.6-Sol-joybuilder --limit 100
"""
import os
import sys
import json
import glob
import time
import socket
import asyncio
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from mas_framework.llm.llm_registry import LLMRegistry
from mas_framework.tools.coding.python_executor import PyExecutor
from experiment.run_manifest import write_manifest
from datasets.gsm8k_dataset import (
    gsm_data_process, multiarith_data_process, svamp_data_process, gsm_get_predict
)
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from datasets.mbpp_dataset import mbpp_data_process
from datasets.math_dataset import math_data_process, math_get_predict, math_check_correctness


def parse_args():
    p = argparse.ArgumentParser(description="Single-agent baseline evaluation")
    p.add_argument('--domain', type=str, default='gsm8k',
                   help='gsm8k | aqua | multiarith | svamp | humaneval')
    p.add_argument('--llm_name', type=str, default='GPT-5.6-Sol-joybuilder')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--limit', type=int, default=None,
                   help='limit number of questions (None = all)')
    return p.parse_args()


def load_dataset(domain):
    if domain == 'gsm8k':
        raw = [json.loads(l) for l in open('datasets/gsm8k/gsm8k.jsonl', 'r', encoding='utf-8')]
        ds = gsm_data_process(raw)
        predict = gsm_get_predict
        check = lambda p, t: float(p) == float(t)
    elif domain == 'multiarith':
        raw = json.load(open('datasets/MultiArith/MultiArith.json', 'r', encoding='utf-8'))
        ds = multiarith_data_process(raw)
        predict = gsm_get_predict
        check = lambda p, t: float(p) == float(t)
    elif domain == 'svamp':
        raw = json.load(open('datasets/SVAMP/SVAMP.json', 'r', encoding='utf-8'))
        ds = svamp_data_process(raw)
        predict = gsm_get_predict
        check = lambda p, t: float(p) == float(t)
    elif domain == 'aqua':
        raw = [json.loads(l) for l in open('datasets/AQuA/AQuA.jsonl', 'r', encoding='utf-8')]
        ds = aqua_data_process(raw)
        predict = aqua_get_predict
        check = lambda p, t: p == t
    elif domain == 'humaneval':
        raw = [json.loads(l) for l in open('datasets/humaneval/humaneval-py.jsonl', 'r', encoding='utf-8')]
        ds = raw  # keep raw records (need prompt + test)
        predict = None  # handled via PyExecutor
        check = None
    elif domain == 'mbpp':
        raw = [json.loads(l) for l in open('datasets/mbpp/mbpp.jsonl', 'r', encoding='utf-8')]
        ds = mbpp_data_process(raw)
        predict = None  # handled via PyExecutor
        check = None
    elif domain == 'math':
        raw = [json.loads(l) for l in open('datasets/math/math.jsonl', 'r', encoding='utf-8')]
        ds = math_data_process(raw)
        predict = math_get_predict
        check = math_check_correctness
    else:
        raise ValueError(f"Unknown domain: {domain}")
    # Align with the full ARG-Designer method: only evaluate the SAME test split
    # that cold_start produced (test_indices), so the comparison is fair.
    ds = slice_test_set(domain, ds)
    return ds, predict, check


def slice_test_set(domain, ds):
    """Restrict ds to test_indices from stage1's task_split_<domain>.json."""
    split_path = os.path.join(
        'output', domain,
        'stage1_cold_start_*', f'task_split_{domain}.json'
    )
    candidates = glob.glob(split_path)
    if not candidates:
        # No split yet — fall back to full set (nothing to align against).
        return ds
    # Pick the most recent stage1 split.
    split_file = sorted(candidates)[-1]
    task_split = json.load(open(split_file, 'r', encoding='utf-8'))
    test_indices = task_split.get('test_indices', [])
    return [ds[i] for i in test_indices]


# Prompt, matching MathSolver's single-agent "math expert" constraint
SYSTEM_PROMPT = (
    "You are a math expert. You will be given a math problem. "
    "Give your own solving process step by step. "
    "The last line of your output contains only the final result without any units, "
    "for example: The answer is 140"
)
AQUA_SYSTEM_PROMPT = (
    "You are a math expert. You will be given a multiple-choice math problem. "
    "Give your own solving process step by step. "
    "The last line of your output contains only the final choice as a capital letter, "
    "for example: The answer is A"
)
CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write a complete function to solve the task. "
    "Give only the Python code, nothing else."
)


async def main():
    args = parse_args()
    started_at = time.time()
    ds, predict, check = load_dataset(args.domain)
    if args.limit:
        ds = ds[:args.limit]
    total = len(ds)
    print(f"Evaluating single-agent baseline on {args.domain}: {total} questions")

    llm = LLMRegistry.get(args.llm_name)

    if args.domain in ('humaneval', 'mbpp'):
        system = CODE_SYSTEM_PROMPT
        executor = PyExecutor()
    elif args.domain == 'aqua':
        system = AQUA_SYSTEM_PROMPT
    else:
        system = SYSTEM_PROMPT

    solved = 0
    results = []
    # chunk into batches to bound concurrency (avoid LLM rate limits)
    for i in range(0, total, args.batch_size):
        batch = ds[i:i + args.batch_size]
        coros = []

        if args.domain in ('humaneval', 'mbpp'):
            for rec in batch:
                msgs = [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': rec['task'] if 'task' in rec else rec['prompt']},
                ]
                coros.append(llm.agen(msgs))
            outs = await asyncio.gather(*coros, return_exceptions=True)

            for rec, out in zip(batch, outs):
                if isinstance(out, Exception):
                    results.append({'task_id': rec.get('task_id'), 'error': str(out),
                                    'correct': False})
                    continue
                code = out.lstrip('```python\n').rstrip('\n```')
                is_solved, _, _ = executor.execute(code, [rec['test']], timeout=10)
                if is_solved:
                    solved += 1
                results.append({'task_id': rec.get('task_id'),
                                'is_solved': is_solved, 'code': code})
        else:
            for item in batch:
                msgs = [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': item['task']},
                ]
                coros.append(llm.agen(msgs))
            outs = await asyncio.gather(*coros, return_exceptions=True)

            for item, out in zip(batch, outs):
                if isinstance(out, Exception):
                    results.append({'question': item['task'], 'pred': None,
                                    'gold': item['answer'], 'correct': False, 'error': str(out)})
                    continue
                pred = predict(out)
                gold = item['answer']
                try:
                    correct = check(pred, gold)
                except Exception:
                    correct = False
                if correct:
                    solved += 1
                results.append({'question': item['task'], 'pred': pred,
                                'gold': gold, 'correct': correct, 'answer': out})

        acc = solved / min(total, i + len(batch)) * 100
        tqdm.write(f"progress {min(total, i+len(batch))}/{total}, acc {acc:.2f}%")

    pass1 = solved / total * 100
    print("\n" + "=" * 50)
    print(f"Single-agent baseline on {args.domain}:")
    print(f"  Total: {total}, Solved: {solved}, Pass@1: {pass1:.2f}%")
    print("=" * 50)

    out_dir = os.path.join('output', args.domain, 'baseline_single_agent')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"baseline_{int(time.time())}.jsonl")
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Results saved to: {out_file}")

    # Write a summary manifest so the run is self-documenting.
    manifest = {
        "stage": "baseline_single_agent",
        "domain": args.domain,
        "llm_name": args.llm_name,
        "hostname": socket.gethostname(),
        "git_head": "unknown",
        "total": total,
        "solved": solved,
        "pass_at_1": pass1,
        "started_at": started_at,
        "finished_at": time.time(),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "argv": sys.argv,
        "args": vars(args),
    }
    manifest_path = write_manifest(out_dir, "baseline", manifest)
    print(f"Manifest written to: {manifest_path}")


if __name__ == '__main__':
    asyncio.run(main())