"""DyLAN for code tasks (humaneval / mbpp), written directly for this repo.

Follows DyLAN's core "confidence aggregation" idea: multiple agents each write
an implementation, they score each other's code (1-5), edges are weighted by
those scores, and a consensus check (>2/3 agreement via the hidden test) lets it
stop early. Final answer is a passing implementation.

Unlike the original DyLAN-HumanEval, correctness is judged objectively by
PyExecutor running the hidden test (not sacrebleu text similarity, not
`cut_def_question` mangling). This makes it work for both humaneval (prompt has
`def ...`) and mbpp (prose + "Function signature: def ...").
"""
import re


class DyLAN_Code:
    def __init__(self, llm_name, num_agents=4, num_rounds=3, seed=0):
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.seed = seed

    # ---- helpers ----
    @staticmethod
    def _extract_entry_point(text):
        m = re.search(r'\bdef\s+(\w+)\s*\(', text)
        return m.group(1) if m else None

    @staticmethod
    def _extract_signature(text):
        m = re.search(r'(def\s+\w+\s*\(.*?\)\s*:)', text)
        return m.group(1) if m else None

    @staticmethod
    def _strip_code_block(s):
        s = (s or "").strip()
        # remove markdown code fences if present
        s = re.sub(r'^```(?:python)?\s*', '', s)
        s = re.sub(r'\s*```$', '', s)
        return s


async def run_dylan_code(llm, query, test, num_agents=4, num_rounds=3, seed=0):
    """Run DyLAN aggregation for one code task; returns (is_solved, code)."""
    from mas_framework.tools.coding.python_executor import PyExecutor
    executor = PyExecutor()
    random.seed(seed)

    entry = DyLAN_Code._extract_entry_point(query)
    # signature line: for mbpp it's "Function signature: def foo(...):", for humaneval it's "def foo(...):"
    sig = DyLAN_Code._extract_signature(query)

    # Round 1: each agent writes an implementation independently.
    codes = []
    for _ in range(num_agents):
        r = await llm.agen([
            {"role": "system", "content": "You are a Python programmer. Write a complete function."},
            {"role": "user", "content": query},
        ])
        c = DyLAN_Code._strip_code_block(r)
        codes.append(c)

    # A code is "correct" if the full program (signature + body) passes the hidden test.
    def full_code(body):
        return (sig + "\n" + body) if sig else body

    def passes(body):
        ok, _, _ = executor.execute(full_code(body), [test], timeout=10)
        return ok

    for round_ in range(1, num_rounds):
        # consensus: >2/3 passing -> stop
        passing = [c for c in codes if passes(c)]
        if len(passing) > (2 / 3) * num_agents:
            break
        # next round: each agent sees others' code + scores (1-5), then revises.
        new_codes = []
        for i in range(num_agents):
            others = [codes[j] for j in range(num_agents) if j != i]
            body = query + "\n\nThese are the implementations from other agents:\n"
            for j, o in enumerate(others, 1):
                body += f"Agent impl {j}: ```python\n{o}\n```\n"
            body += (
                "\nUsing the reasoning from other agents, give an updated complete implementation. "
                "Output ONLY the python code. Also give a score 1-5 to each other impl in [[...]] form."
            )
            r = await llm.agen([
                {"role": "system", "content": "You are a Python programmer. Write a complete function."},
                {"role": "user", "content": body},
            ])
            new_codes.append(DyLAN_Code._strip_code_block(r))
        codes = new_codes

    # pick a passing implementation (prefer the first that passes)
    for c in codes:
        if passes(c):
            return True, c
    return False, (codes[0] if codes else "")