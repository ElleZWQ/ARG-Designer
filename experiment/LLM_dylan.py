"""DyLAN baseline (math version): Dynamic LLM Agent Network, rewritten for this repo.

Same role as `LLM_debate` — a "confidence aggregation" baseline — but adds
DyLAN's dynamic machinery: agents score each other's answers (1-5), those scores
update edge weights, and a consensus check (>2/3 agreement) lets it stop early.

We drop the original MAS/vLLM coupling and reuse this repo's GPT-5.6-Sol LLM
calls plus the existing math answer checkers (gsm_get_predict / math_check_correctness).
"""
import asyncio
import random
import re

from mas_framework.llm.llm_registry import LLMRegistry


class DyLAN_Math:
    def __init__(self, llm_name, num_agents=3, num_rounds=3, seed=0):
        self.llm = LLMRegistry.get(llm_name)
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.seed = seed
        # system prompt mirroring DyLAN's math debate
        self.system_prompt = (
            "It's a debate. Explain your reasons at each round thoroughly and give "
            "the final answer at the end."
        )

    # ---- answer extraction: reuse repo checkers where possible, keep a light local fallback ----
    @staticmethod
    def _extract_answer(text):
        # reuse the repo's gsm answer extractor (handles boxed, fractions, numbers)
        try:
            from datasets.gsm8k_dataset import gsm_get_predict
            a = gsm_get_predict(text)
            if a:
                return a
        except Exception:
            pass
        # fallback: last "The answer is ..." / boxed
        m = re.findall(r"\\boxed\{([^}]*)\}", text)
        if m:
            return m[-1].strip()
        m = re.findall(r"answer is[:\s]*([-\d\./]+)", text, re.IGNORECASE)
        return m[-1].strip() if m else ""

    @staticmethod
    def _equiv(a, b):
        import string as _s
        def norm(x):
            x = (x or "").strip().lower()
            x = x.replace(" ", "").replace("\\", "")
            return x
        return norm(a) == norm(b)

    def _most_frequent(self, answers):
        if not answers:
            return "", 0
        best, best_cnt = answers[0], 0
        for a in answers:
            cnt = sum(self._equiv(a, b) for b in answers)
            if cnt > best_cnt:
                best, best_cnt = a, cnt
        return best, best_cnt

    def _consensus_answer(self, contexts):
        answers = [self._extract_answer(c[-1]["content"]) for c in contexts]
        answers = [a for a in answers if a]
        if not answers:
            return "", 0
        ans, cnt = self._most_frequent(answers)
        return ans, cnt

    def _score_prompt(self, question, others):
        """Build the second-round message asking for an updated answer + a score for each other answer."""
        body = f"{question}\n\nThese are the solutions from other agents:\n"
        for i, o in enumerate(others, 1):
            body += f"Agent solution {i}: ```{o}```\n"
        body += (
            "\nUsing the reasoning from other agents as additional advice with critical "
            "thinking, give an updated answer. Along with the answer, give a score from 1 to 5 "
            "to each of the other solutions, in the form [[1, 5, 2, ...]].\n"
        )
        return body

    async def run(self, query):
        random.seed(self.seed)
        # round 1: each agent answers independently
        contexts = []
        for _ in range(self.num_agents):
            r = await self.llm.agen([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ])
            contexts.append([{"role": "system", "content": self.system_prompt},
                             {"role": "assistant", "content": r}])

        for rid in range(1, self.num_rounds):
            # consensus check: if > 2/3 agree, stop early
            ans, cnt = self._consensus_answer(contexts)
            if cnt > (2 / 3) * self.num_agents:
                break

            # next round: each agent sees all others' answers and scores them
            new_contexts = []
            for i in range(self.num_agents):
                others = [contexts[j][-1]["content"] for j in range(self.num_agents) if j != i]
                prompt = self._score_prompt(query, others)
                r = await self.llm.agen([
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ])
                new_contexts.append([{"role": "system", "content": self.system_prompt},
                                     {"role": "assistant", "content": r}])
            contexts = new_contexts

        ans, _ = self._consensus_answer(contexts)
        return ans if ans else ""


async def run_dylan(llm_name, query, num_agents=3, num_rounds=3, seed=0):
    m = DyLAN_Math(llm_name, num_agents=num_agents, num_rounds=num_rounds, seed=seed)
    return await m.run(query)