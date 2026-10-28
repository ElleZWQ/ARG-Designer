"""DyLAN-HumanEval baseline, rewritten for this repo (no MAS / vLLM).

Direct port of the original DyLAN_HumanEval (AgentArk) with the MAS base class
removed; all LLM calls go through the repo's GPT gateway (LLMRegistry). The
agent+judge multi-role graph, scoring, edge weights, consensus, and sacrebleu
similarity logic are preserved from `dylan_humaneval_utils`.

Usage (via baseline.py --method dylan on humaneval / mbpp).
"""
import ast
import math
import random
import re

from sacrebleu import sentence_bleu

from mas_framework.llm.llm_registry import LLMRegistry
from experiment.dylan_humaneval_utils import (
    ROLE_MAP, ROLE_MAP_INIT, JUDGE_MAP, JUDGE_PREFIX, JUDGE_NAMES,
    CODE_THRESHOLD, TOOL_LIST,
    parse_code_completion, check_function_result, parse_ranks, most_frequent,
)


class DyLAN_HumanEval:
    def __init__(self, llm_name, num_agents=4, num_judges=4, num_rounds=3, seed=0):
        self.llm = LLMRegistry.get(llm_name)
        self.seed = seed
        self.num_agents = num_agents
        self.num_judges = num_judges
        self.num_rounds = num_rounds
        self.activation = "listwise"

        self.agent_roles = [
            "PythonAssistant", "AlgorithmDeveloper", "ComputerScientist", "Programmer",
        ][:num_agents]
        if len(self.agent_roles) != num_agents:
            self.agent_roles += ["PythonAssistant"] * (num_agents - len(self.agent_roles))

        self.judge_roles = ["Tester", "Reflector", "Debugger", "QualityManager"][:num_judges]
        if len(self.judge_roles) != num_judges:
            self.judge_roles += ["Tester"] * (num_judges - len(self.judge_roles))

        self.nodes = []
        self.edges = []

    # ---- helpers (same as original) ----
    def _call(self, messages):
        return self.llm.gen(messages)

    def parse_human_eval_query(self, query):
        entry_point = None
        prompt = query
        for line in query.split("\n"):
            if line.startswith("def "):
                ep = line.split("def ")[1].split("(")[0].strip()
                if ep:
                    entry_point = ep
                    break
        return prompt, entry_point

    def construct_message(self, responses, question):
        if len(responses) == 0:
            qtemplate = "```python\n{}\n```".format(question)
            qtemplate = (
                "You must complete the python function I give you.\n"
                "Be sure to use the same indentation I specified. Furthermore, you may "
                "only write your response in code/comments.\n[function impl]:\n{}\n"
                "Once more, please follow the template by repeating the original function, "
                "then writing the completion.".format(qtemplate)
            )
            return {"role": "user", "content": qtemplate}
        else:
            flag_ranker = False
            ranker = None
            for agent in responses:
                if agent["role"] == "Ranker":
                    flag_ranker = True
                    ranker = agent
                    break
            pre_impls = list(responses[0]["answer"].keys())
            if flag_ranker:
                ranks = ranker["answer"]
                pre_impls = [p for p in pre_impls if ranks[p]]

            prefix_string = ""
            for impl_id, pre_impl in enumerate(pre_impls, start=1):
                prefix_string += "[previous impl {}]:\n```python\n{}\n```\n\n".format(impl_id, pre_impl)
                for agent in responses:
                    if agent["role"] == "Ranker":
                        continue
                    reply = agent["answer"][pre_impl]
                    prefix_string += JUDGE_PREFIX[agent["role"]].format(impl_id) + reply + "\n\n"

            def connect_judges(judges):
                judges = [j for j in judges if j["role"] != "Ranker"]
                if len(judges) == 0:
                    raise ValueError("No judges")
                if len(judges) == 1:
                    return JUDGE_NAMES[judges[0]["role"]]
                if len(judges) == 2:
                    return JUDGE_NAMES[judges[0]["role"]] + " and " + JUDGE_NAMES[judges[1]["role"]]
                return ", ".join([JUDGE_NAMES[j["role"]] for j in judges[:-1]]) + " and " + JUDGE_NAMES[judges[-1]["role"]]

            prefix_string += (
                "You must complete the python function I give you by rectifying previous "
                "implementations. Use the other information as a hint.\n"
                "Be sure to use the same indentation I specified. Furthermore, you may only "
                "write your response in code/comments.\n[improved impl]:\n```python\n{}\n```\n\n"
                "Please follow the template by repeating the function signature and complete the "
                "new implementation in [improved impl]. If no changes are needed, simply rewrite "
                "the implementation in the Python code block.\n"
                "Along with the new implementation, give a score ranged from 1 to 5 to {} in terms "
                "of helpfulness. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of "
                "your response.".format(question, connect_judges(responses), len(responses) - 1)
            )
            return {"role": "user", "content": prefix_string}

    def construct_judge_message(self, responses, question, role):
        if role == "Tester":
            prefix_string = "function signature:\n```python\n{}\n```\n\nunit tests:\n".format(question)
            return {"role": "user", "content": prefix_string}
        elif role == "Reflector":
            prefix_string = (
                "Here are previous implementations of the same function.The function has a "
                "signature and a docstring explaining its functionality.\n\n"
            )
            for aid, ar in enumerate(responses, start=1):
                prefix_string += "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, ar)
            prefix_string += (
                "Write your reflection on these implementations in consideration of correctness, "
                "efficiency, and possible corner cases. Put your reflection of the n-th implementation "
                "after [reflection n].\nAlong with the reflections, give a score ranged from 1 to 5 to "
                "each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at "
                "the end of your response.".format(len(responses))
            )
            return {"role": "user", "content": prefix_string}
        elif role == "Debugger":
            prefix_string = (
                "Here are previous implementations of the same function.The function has a "
                "signature and a docstring explaining its functionality.\n\n"
            )
            for aid, ar in enumerate(responses, start=1):
                prefix_string += "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, ar)
            prefix_string += (
                "Debug this version of implementation and write your feedback as a debugger. Put your "
                "debug information of the n-th implementation after [bug fix n].\nPlease give a score "
                "ranged from 1 to 5 to each previous implementation. Put all {} scores in the form like "
                "[[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            )
            return {"role": "user", "content": prefix_string}
        elif role == "QualityManager":
            prefix_string = (
                "Here are previous implementations of the same function.The function has a "
                "signature and a docstring explaining its functionality.\n\n"
            )
            for aid, ar in enumerate(responses, start=1):
                prefix_string += "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, ar)
            prefix_string += (
                "Write your code review on these implementations in multiple aspects. Put your review "
                "of the n-th implementation after [code review n].\nAlso, give a score ranged from 1 to "
                "5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] "
                "at the end of your response.".format(len(responses))
            )
            return {"role": "user", "content": prefix_string}
        elif role == "Ranker":
            prefix_string = (
                "Here are some implementations of the same function. The function has a signature and "
                "a docstring explaining its functionality.\n\n"
            )
            for aid, ar in enumerate(responses, start=1):
                prefix_string += "[function impl {}]:\n```python\n{}\n```\n\n".format(aid, ar)
            prefix_string += (
                "[function signature]:\n```python\n{}\n```\n\nTake correctness, efficiency, and possible "
                "corner cases into consideration, choose top 2 solutions that match the function's docstring "
                "best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of "
                "your response.".format(question)
            )
            return {"role": "user", "content": prefix_string}
        else:
            raise NotImplementedError(f"Unknown role: {role}")

    def construct_ranking_message(self, responses, question):
        prefix_string = (
            "Here are some implementations of the same function and the thoughts of them. The function "
            "has a signature and a docstring explaining its functionality.\n\n"
        )
        for aid, ar in enumerate(responses, start=1):
            prefix_string += "[function impl {}]:\n```python\n{}\n```\n\n".format(aid, ar)
        prefix_string += (
            "[function signature]:\n```python\n{}\n```\n\nTake correctness, efficiency, and possible "
            "corner cases into consideration, choose top 2 solutions that match the function's docstring "
            "best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of "
            "your response.".format(question)
        )
        return {"role": "user", "content": prefix_string}

    def listwise_ranker_2(self, responses, question):
        message = self.construct_ranking_message(responses, question)
        completion = self._call([message])
        return parse_ranks(completion, max_num=len(responses))

    # ---- node/edge management ----
    def init_nodes(self):
        self.nodes = []
        for rid in range(self.num_rounds):
            for idx, role in enumerate(self.agent_roles):
                node_id = rid * (self.num_agents + self.num_judges) + idx
                self.nodes.append({
                    "id": node_id, "type": "agent", "role": role,
                    "reply": None, "answer": "", "active": False, "question": None,
                    "from_edges": [], "to_edges": [], "importance": 0,
                })
            if rid < self.num_rounds - 1:
                for idx, role in enumerate(self.judge_roles):
                    node_id = rid * (self.num_agents + self.num_judges) + self.num_agents + idx
                    self.nodes.append({
                        "id": node_id, "type": "judge", "role": role,
                        "reply": None, "answer": "", "active": False, "question": None,
                        "from_edges": [], "to_edges": [], "importance": 0,
                        "unit_tests": [],
                    })

        self.edges = []
        for rid in range(self.num_rounds - 1):
            agent_offset = rid * (self.num_agents + self.num_judges)
            judge_offset = agent_offset + self.num_agents
            next_agent_offset = (rid + 1) * (self.num_agents + self.num_judges)
            for agent_idx in range(self.num_agents):
                agent_id = agent_offset + agent_idx
                for judge_idx in range(self.num_judges):
                    judge_id = judge_offset + judge_idx
                    self.edges.append({"from": agent_id, "to": judge_id, "weight": 0})
                    self.nodes[agent_id]["to_edges"].append(len(self.edges) - 1)
                    self.nodes[judge_id]["from_edges"].append(len(self.edges) - 1)
            for judge_idx in range(self.num_judges):
                judge_id = judge_offset + judge_idx
                for agent_idx in range(self.num_agents):
                    next_agent_id = next_agent_offset + agent_idx
                    self.edges.append({"from": judge_id, "to": next_agent_id, "weight": 0})
                    self.nodes[judge_id]["to_edges"].append(len(self.edges) - 1)
                    self.nodes[next_agent_id]["from_edges"].append(len(self.edges) - 1)

    def deactivate_all_nodes(self):
        for node in self.nodes:
            node["active"] = False
            node["reply"] = None
            node["answer"] = ""
            node["question"] = None
            node["importance"] = 0
            if node["type"] == "judge" and node["role"] == "Tester":
                node["unit_tests"] = []

    def zero_grad(self):
        for edge in self.edges:
            edge["weight"] = 0

    def find_array(self, text):
        matches = re.findall(r'\[\[(.*?)\]\]', text)
        if matches:
            last_match = matches[-1].replace(' ', '')
            try:
                return list(map(int, last_match.split(',')))
            except Exception:
                return []
        return []

    def _update_weights(self, shuffled_idxs, weights, formers, original_idxs):
        pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(pairs, key=lambda x: original_idxs.index(x[0]))
        weights = [w for _, w, _ in sorted_pairs]
        formers = [(f, eid) for eid, _, f in sorted_pairs]
        for (_, eid), w in zip(formers, weights):
            self.edges[eid]["weight"] = w / 5 if 0 < w <= 5 else (1 if w > 5 else 0)
        total = sum(self.edges[eid]["weight"] for _, eid in formers)
        if total > 0:
            for _, eid in formers:
                self.edges[eid]["weight"] /= total
        else:
            for _, eid in formers:
                self.edges[eid]["weight"] = 1 / len(formers)
        return weights, formers

    def activate_llm_node(self, node_id, question):
        node = self.nodes[node_id]
        node["question"] = question
        node["active"] = True
        sys_prompt = ROLE_MAP_INIT[node["role"]] if len(node["from_edges"]) == 0 else ROLE_MAP[node["role"]]
        contexts = [{"role": "system", "content": sys_prompt}]
        formers = []
        for edge_idx in node["from_edges"]:
            edge = self.edges[edge_idx]
            from_node = self.nodes[edge["from"]]
            if from_node["reply"] is not None and from_node["active"]:
                formers.append((from_node, edge_idx))
        original_idxs = [m[1] for m in formers]
        random.shuffle(formers)
        shuffled_idxs = [m[1] for m in formers]
        formers = [m[0] for m in formers]
        contexts.append(self.construct_message(formers, question))
        node["reply"] = self._call(contexts)
        node["answer"] = parse_code_completion(node["reply"], question)
        weights = self.find_array(node["reply"])
        if len(weights) != len(formers) - 1:
            weights = [0 for _ in range(len(formers))]
        else:
            res_weights = [3] if (formers and formers[0]["role"] == "Ranker") else []
            for wid, weight in enumerate(weights):
                res_weights.append(weight)
                if formers and wid + 1 < len(formers) and formers[wid + 1]["role"] == "Ranker":
                    res_weights.append(3)
            weights = res_weights
        self._update_weights(shuffled_idxs, weights, formers, original_idxs)

    def activate_judge_node(self, node_id, question):
        node = self.nodes[node_id]
        node["question"] = question
        node["active"] = True
        sys_prompt = JUDGE_MAP[node["role"]]
        contexts = [{"role": "system", "content": sys_prompt}]
        formers = []
        for edge_idx in node["from_edges"]:
            edge = self.edges[edge_idx]
            from_node = self.nodes[edge["from"]]
            if from_node["reply"] is not None and from_node["active"]:
                formers.append((from_node["answer"], edge_idx))
        original_idxs = [m[1] for m in formers]
        random.shuffle(formers)
        shuffled_idxs = [m[1] for m in formers]
        formers = [m[0] for m in formers]

        if node["role"] == "Ranker" and len(formers) <= 2:
            node["reply"] = "[1, 2]"
        elif node["role"] not in TOOL_LIST:
            contexts.append(self.construct_judge_message(formers, question, node["role"]))
            node["reply"] = self._call(contexts)
        else:
            node["reply"] = formers

        node["answer"] = self.parse_judge_attitude(node["reply"], question, node["role"], formers)
        if node["role"] == "Tester":
            node["answer"], node["unit_tests"] = node["answer"]

        weights = self.find_array_for_judge(node["reply"], node["role"], formers, node["answer"])
        if len(weights) != len(formers):
            weights = [0 for _ in range(len(formers))]
        self._update_weights(shuffled_idxs, weights, formers, original_idxs)

    def find_array_for_judge(self, text, role, formers, answer):
        if role == "Ranker":
            return [5 if answer[f] else 0 for f in formers]
        if role == "Passer":
            return [5 if answer[f] == "The code doesn't have syntax error." else 0 for f in formers]
        if role == "Tester":
            results = []
            for f in formers:
                if "Passed tests:\n" not in answer[f]:
                    results.append(0)
                else:
                    total_tests = len(answer[f].splitlines()) - 1
                    if "Failed tests:\n" in answer[f]:
                        total_tests -= 2
                    flag_pass = False
                    pass_tests = 0
                    for line in answer[f].splitlines():
                        if flag_pass:
                            if "Failed tests:" in line:
                                break
                            pass_tests += 1
                        if "Passed tests:" in line:
                            flag_pass = True
                    pass_tests -= 1
                    results.append(math.ceil(pass_tests / total_tests * 5) if total_tests else 0)
            return results
        # Reflector / Debugger / QualityManager: parse [[...]]
        matches = re.findall(r'\[\[(.*?)\]\]', text)
        if matches:
            last_match = matches[-1].replace(' ', '')
            try:
                return list(map(int, last_match.split(',')))
            except Exception:
                return []
        return []

    def parse_judge_attitude(self, agent_response, question, role, former_results):
        if role == "Passer":
            attitude = {res_code: ("The code doesn't have syntax error." if check_function_result(res_code)["passed"]
                                   else "The code has syntax error: " + check_function_result(res_code)["result"])
                        for res_code in former_results}
            return attitude
        elif role == "Tester":
            def parse_tests(tests):
                candidates = [t.strip() for t in tests.splitlines() if "assert" in t]
                import ast as _ast
                candidates = [t for t in candidates if self._syntax_ok(t)]
                candidates = list(set(candidates))
                return candidates[:10]
            tests = parse_tests(agent_response)
            if len(tests) == 0:
                return ({res_code: "" for res_code in former_results}, [])
            attitude = {}
            for res_code in former_results:
                passed, failed = [], []
                for test in tests:
                    r = check_function_result(res_code + "\n" + test)
                    (passed if r["passed"] else failed).append(test if r["passed"] else test + " # output: error")
                s = ""
                if passed:
                    s += "Passed tests:\n" + "\n".join(passed) + "\n"
                if failed:
                    if s:
                        s += "\n"
                    s += "Failed tests:\n" + "\n".join(failed) + "\n"
                attitude[res_code] = s
            return (attitude, tests)
        elif role == "Reflector":
            reflections = re.findall(r'\[reflection \d+\]:\n(.*?)(?=\[reflection \d+\]:|$)', agent_response, re.DOTALL)
            reflections = [r.strip() for r in reflections]
            if len(reflections) != len(former_results):
                reflections = ["No reflection"] * len(former_results)
            return {rc: r for rc, r in zip(former_results, reflections)}
        elif role == "Debugger":
            reflections = re.findall(r'\[bug fix \d+\]:\n(.*?)(?=\[bug fix \d+\]:|$)', agent_response, re.DOTALL)
            reflections = [r.strip() for r in reflections]
            if len(reflections) != len(former_results):
                reflections = ["No Bugs"] * len(former_results)
            return {rc: r for rc, r in zip(former_results, reflections)}
        elif role == "QualityManager":
            reflections = re.findall(r'\[code review \d+\]:\n(.*?)(?=\[code review \d+\]:|$)', agent_response, re.DOTALL)
            reflections = [r.strip() for r in reflections]
            if len(reflections) != len(former_results):
                reflections = ["No code review"] * len(former_results)
            return {rc: r for rc, r in zip(former_results, reflections)}
        elif role == "Ranker":
            tops = parse_ranks(agent_response, max_num=len(former_results)) if 2 < len(former_results) <= 4 else [0, 1]
            return {rc: (i in tops) for i, rc in enumerate(former_results)}
        else:
            raise NotImplementedError(f"Unknown role: {role}")

    @staticmethod
    def _syntax_ok(code):
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    def cut_def_question(self, func_code, question, entry_point):
        def parse_imports(src):
            res = ["    " + l.strip() for l in src.split("\n") if "import" in l]
            return res
        import_lines = parse_imports(func_code)
        def extract_functions(src):
            tree = ast.parse(src)
            funcs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    parents = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    if sum(1 for p in parents if p.lineno <= node.lineno and p.end_lineno >= node.end_lineno) == 1:
                        funcs.append("\n".join(src.splitlines()[node.lineno - 1:node.end_lineno]))
            return funcs
        try:
            funcs = extract_functions(func_code)
        except Exception:
            funcs = [func_code]
        def extract_func_def(src):
            for line in src.split("\n"):
                if "def" in line and entry_point in line:
                    return line
            return ""
        que_func = extract_func_def(question)
        for fiid, func_ins_code in enumerate(funcs):
            if question in func_ins_code:
                func_ins_code = func_ins_code.split(question)[-1]
            elif question.strip() in func_ins_code:
                func_ins_code = func_ins_code.split(question.strip())[-1]
            elif que_func in func_ins_code:
                res_lines = func_ins_code.split("\n")
                func_ins_code = ""
                in_func = False
                for line in res_lines:
                    if in_func:
                        func_ins_code += line + "\n"
                    if "def" in line:
                        in_func = True
            else:
                continue
            other_funcs = []
            for other_func in funcs[:fiid] + funcs[fiid + 1:]:
                other_func_lines = other_func.split("\n")
                other_func = "\n".join(other_func_lines[:1] + import_lines + other_func_lines[1:])
                other_funcs.append(other_func)
            return "\n".join(import_lines) + "\n" + func_ins_code + "\n" + "\n".join(other_funcs)
        res_lines = func_code.split("\n")
        func_code = ""
        in_func = False
        for line in res_lines:
            if in_func:
                func_code += line + "\n"
            if "def" in line:
                in_func = True
        return "\n".join(import_lines) + "\n" + func_code

    def check_consensus(self, active_nodes, idx_mask, question, entry_point):
        candidates = [node["answer"] for node in active_nodes]
        python_codes, backup = [], []
        for cand in candidates:
            if check_function_result(cand)["passed"]:
                python_codes.append(cand)
            else:
                backup.append(cand)
        if len(python_codes) == 0:
            return False, None
        pred_answers = [self.cut_def_question(pc, question, entry_point) for pc in python_codes]
        cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= CODE_THRESHOLD * 100
        consensus_answer, ca_cnt = most_frequent(pred_answers, cmp_res)
        if ca_cnt > math.floor(2 / 3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def all_tests_and_get_final_result(self, question, unit_tests, entry_point, agent_nodes):
        candidates = [node["answer"] for node in agent_nodes if node["type"] == "agent" and node["active"]]
        candidates = [self.cut_def_question(c, question, entry_point) for c in candidates]
        python_codes = []
        for cand in candidates:
            passed, failed = [], []
            for test in unit_tests:
                r = check_function_result(question + "\n" + cand + "\n" + test)
                (passed if r["passed"] else failed).append(test)
            python_codes.append((cand, len(passed), passed, failed))
        sorted_codes = sorted(python_codes, key=lambda x: x[1], reverse=True)
        max_passed = sorted_codes[0][1] if sorted_codes else 0
        top_codes = [c for c in sorted_codes if c[1] == max_passed]
        if len(top_codes) < 5:
            top_codes = [c for c in sorted_codes if c[1] in (max_passed, max_passed - 1)]
        selected = random.choice(top_codes) if top_codes else (sorted_codes[0] if sorted_codes else (question, 0, [], []))
        return selected[0]

    def inference(self, sample):
        query = sample["query"]
        random.seed(self.seed)
        prompt, entry_point = self.parse_human_eval_query(query)
        if not entry_point:
            for line in prompt.split("\n"):
                if line.startswith("def "):
                    ep = line.split("def ")[1].split("(")[0].strip()
                    if ep:
                        entry_point = ep
                        break
        if not entry_point:
            raise ValueError("Could not determine entry point for the function")

        self.init_nodes()
        self.zero_grad()
        self.deactivate_all_nodes()

        unit_tests = []
        loop_indices = list(range(self.num_agents))
        random.shuffle(loop_indices)
        for node_idx in loop_indices:
            self.activate_llm_node(node_idx, prompt)

        for node_idx in range(self.num_agents, self.num_agents + self.num_judges):
            self.activate_judge_node(node_idx, prompt)
            if self.nodes[node_idx]["role"] == "Tester":
                unit_tests.extend(self.nodes[node_idx]["unit_tests"])

        loop_indices = list(range(self.num_agents + self.num_judges, self.num_agents * 2 + self.num_judges))
        random.shuffle(loop_indices)
        activated_agent_indices = []
        for idx, node_idx in enumerate(loop_indices):
            self.activate_llm_node(node_idx, prompt)
            activated_agent_indices.append(node_idx)
            if idx >= math.floor(2 / 3 * self.num_agents):
                active_nodes = [self.nodes[i] for i in activated_agent_indices]
                reached, _ = self.check_consensus(active_nodes, list(range(self.num_agents)), prompt, entry_point)
                if reached:
                    agent_nodes = [n for n in self.nodes if n["type"] == "agent" and n["active"]]
                    return {'response': self.all_tests_and_get_final_result(prompt, unit_tests, entry_point, agent_nodes)}

        idx_mask = list(range(self.num_agents))
        active_agent_indices = list(range(self.num_agents + self.num_judges, self.num_agents * 2 + self.num_judges))
        for rid in range(2, self.num_rounds):
            if self.num_agents > 2:
                replies = [self.nodes[idx]["answer"] for idx in active_agent_indices]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[i] for i in indices]
                tops = self.listwise_ranker_2(shuffled_replies, prompt)
                idx_mask = [active_agent_indices[indices[x]] % (self.num_agents + self.num_judges) for x in tops]
            judge_start_idx = (self.num_agents + self.num_judges) * rid - self.num_judges
            for node_idx in range(judge_start_idx, judge_start_idx + self.num_judges):
                self.activate_judge_node(node_idx, prompt)
                if self.nodes[node_idx]["role"] == "Tester":
                    unit_tests.extend(self.nodes[node_idx]["unit_tests"])
            agent_start_idx = (self.num_agents + self.num_judges) * rid
            loop_indices = list(range(agent_start_idx, agent_start_idx + self.num_agents))
            random.shuffle(loop_indices)
            active_agent_indices = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    self.activate_llm_node(node_idx, prompt)
                    active_agent_indices.append(node_idx)
                    if len(active_agent_indices) > math.floor(2 / 3 * len(idx_mask)):
                        active_nodes = [self.nodes[i] for i in active_agent_indices]
                        reached, _ = self.check_consensus(active_nodes, idx_mask, prompt, entry_point)
                        if reached:
                            agent_nodes = [n for n in self.nodes if n["type"] == "agent" and n["active"]]
                            return {'response': self.all_tests_and_get_final_result(prompt, unit_tests, entry_point, agent_nodes)}

        agent_nodes = [n for n in self.nodes if n["type"] == "agent" and n["active"]]
        return {'response': self.all_tests_and_get_final_result(prompt, unit_tests, entry_point, agent_nodes)}