"""LLM-Debate baseline (Du et al. 2023), rewritten for this repo.

The official implementation (https://github.com/composable-models/llm_multiagent_debate)
relies on its own `MAS` base class and vLLM. Here we drop both and drive the
debate with the repo's existing LLM gateway (GPT-5.6-Sol-joybuilder via
`mas_framework.llm`), keeping only the multi-agent debate algorithm itself.

Agent count is chosen per dataset to match ARG-Designer's typical topology
size; see `AGENT_NUM`.
"""
import os
import time
import asyncio

from mas_framework.llm.llm_registry import LLMRegistry

# Agent count per dataset, matching ARG-Designer's typical topology breadth.
# mmlu is omitted (content-filtered, not evaluated here).
AGENT_NUM = {
    "gsm8k": 3,
    "aqua": 3,
    "math": 3,
    "multiarith": 3,
    "svamp": 3,
    "humaneval": 4,
    "mbpp": 4,
}

LLM_NAME = os.getenv("LLM_NAME", "GPT-5.6-Sol-joybuilder")


class LLM_Debate_Main:
    """Multi-agent debate over a single LLM backend (no vLLM, no MAS)."""

    def __init__(self, rounds_num=2, agents_num=None, dataset_name=None):
        self.rounds_num = rounds_num
        # agents_num is per-dataset; resolved at inference time if not fixed.
        self._fixed_agents = agents_num
        self.dataset_name = dataset_name
        self.llm = LLMRegistry.get(LLM_NAME)

    def _agents_num(self, dataset_name=None):
        if self._fixed_agents is not None:
            return self._fixed_agents
        ds = dataset_name or self.dataset_name
        return AGENT_NUM.get(ds, 5)

    async def _call(self, messages):
        """Call the LLM with a list of {role, content} dicts; return text.

        Wrap in a timeout so a single hung request can't stall the whole run.
        """
        return await asyncio.wait_for(self.llm.agen(messages), timeout=120.0)

    @staticmethod
    def _as_dict(content):
        return {"role": "assistant", "content": content}

    def construct_message(self, agents, question, idx):
        if len(agents) == 0:
            return {
                "role": "user",
                "content": (
                    "Can you verify that your answer is correct. Please reiterate "
                    "your answer, making sure to state your answer at the end of "
                    "the response."
                ),
            }

        prefix_string = "# These are the recent/updated opinions from other agents: "
        for agent in agents:
            agent_response = agent[idx]["content"]
            prefix_string += (
                "\n# ONE AGENT RESPONSE\n```\n"
                f"{agent_response}\n```\n----- END OF THE AGENT RESPONSE -----\n"
            )
        prefix_string += (
            "\nUse these opinions carefully as additional advice, can you provide "
            "an updated answer? Make sure to state your answer at the end of the "
            "response.\n"
        )
        # Skip QMSum/QASPER/HotpotQA special-casing; always include the question.
        prefix_string += (
            "\n----- THE ORIGINAL PROBLEM -----\n"
            f"{question}\n"
            "----- END OF THE ORIGINAL PROBLEM -----\n"
        )
        return {"role": "user", "content": prefix_string}

    def _aggregate_prompt(self, query, answers):
        prompt = f"Task:\n{query}\n\n"
        for i, answer in enumerate(answers):
            prompt += f"Solution {i + 1}:\n{answer}\n\n"
        prompt += (
            "Given all the above solutions, reason over them carefully and provide "
            "a final answer to the task. Please put your final answer in the format "
            "of Final Answer: "
        )
        return prompt

    async def inference_async(self, sample, dataset_name=None):
        """Debate one sample; returns {'response': <final answer>}."""
        query = sample["query"]
        n = self._agents_num(dataset_name)

        agent_contexts = [
            [
                {
                    "role": "user",
                    "content": (
                        f"{query} Make sure to state your answer at the end of the response."
                    ),
                }
            ]
            for _ in range(n)
        ]

        for round_ in range(self.rounds_num):
            for i, agent_context in enumerate(agent_contexts):
                if round_ != 0:
                    others = agent_contexts[:i] + agent_contexts[i + 1 :]
                    msg = self.construct_message(others, query, 2 * round_ - 1)
                    agent_context.append(msg)
                response = await self._call(agent_context)
                agent_context.append(self._as_dict(response))

        answers = [c[-1]["content"] for c in agent_contexts]
        agg_prompt = self._aggregate_prompt(query, answers)
        final_answer = await self._call([{"role": "user", "content": agg_prompt}])

        formatted = "".join(
            f"===== Solution {i + 1} =====\n{a}\n" for i, a in enumerate(answers)
        )
        return {
            "response": formatted + f"===== Solution {n + 1} =====\n{final_answer}"
        }

    async def inference_batch(self, samples, dataset_name=None):
        """Debate all samples (rounds sequential, agents parallel in a round)."""
        n = self._agents_num(dataset_name)
        # Initialize contexts: (sample_idx, agent_idx) -> list of messages.
        all_agent_contexts = [
            [
                [
                    {
                        "role": "user",
                        "content": (
                            f"{s['query']} Make sure to state your answer at the end of the response."
                        ),
                    }
                ]
                for _ in range(n)
            ]
            for s in samples
        ]

        for round_ in range(self.rounds_num):
            prompts = []
            meta = []
            for sample_idx, ctxs in enumerate(all_agent_contexts):
                for agent_idx, ctx in enumerate(ctxs):
                    if round_ != 0:
                        others = ctxs[:agent_idx] + ctxs[agent_idx + 1 :]
                        msg = self.construct_message(
                            others,
                            samples[sample_idx]["query"],
                            2 * round_ - 1,
                        )
                        ctx.append(msg)
                    prompts.append(ctx)
                    meta.append((sample_idx, agent_idx))

            t0 = time.time()
            responses = await asyncio.gather(*[self._call(p) for p in prompts])
            t1 = time.time()
            print(f"Round {round_ + 1}/{self.rounds_num} inference: {t1 - t0:.2f}s")

            for resp, (sample_idx, agent_idx) in zip(responses, meta):
                all_agent_contexts[sample_idx][agent_idx].append(self._as_dict(resp))

        # Aggregate all samples.
        agg_prompts = []
        for s, ctxs in zip(samples, all_agent_contexts):
            answers = [c[-1]["content"] for c in ctxs]
            agg_prompts.append(self._aggregate_prompt(s["query"], answers))

        t0 = time.time()
        agg_responses = await asyncio.gather(
            *[self._call([{"role": "user", "content": p}]) for p in agg_prompts]
        )
        t1 = time.time()
        print(f"Aggregation inference: {t1 - t0:.2f}s")

        results = []
        for s, ctxs, final in zip(samples, all_agent_contexts, agg_responses):
            answers = [c[-1]["content"] for c in ctxs]
            formatted = "".join(
                f"===== Solution {i + 1} =====\n{a}\n" for i, a in enumerate(answers)
            )
            results.append(
                {"response": formatted + f"===== Solution {n + 1} =====\n{final}"}
            )
        return results