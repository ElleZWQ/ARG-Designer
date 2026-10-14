import re


def extract_sig(code: str) -> str:
    """Return the first def/class line from the reference solution.

    MBPP only gives a natural-language `text` (no function signature), so the
    model otherwise invents its own function name and fails the asserts. We add
    the reference signature so the generated function matches the test_list.
    """
    if not code:
        return ""
    for line in code.strip().split("\n"):
        line = line.strip()
        if line.startswith("def ") or line.startswith("class "):
            return line
    return ""


def mbpp_data_process(dataset):
    """Convert raw MBPP records into {task, test} rows.

    Each MBPP record has: text (natural-language task), test_list (list of
    assert statements), and code (reference solution). We build a single test
    string by joining test_list (matching how HumanEval stores its `test`
    field), and prepend the reference function signature to `task` so the model
    uses the correct function name/signature.
    """
    out = []
    for data in dataset:
        test_list = data.get('test_list') or []
        test = "\n".join(test_list)
        if data.get('test_setup_code'):
            test = data['test_setup_code'] + "\n" + test

        task = data["text"]
        sig = extract_sig(data.get('code', ''))
        if sig:
            task = f"{task}\nFunction signature: {sig}\n"

        item = {
            "task": task,
            "test": test,
            "task_id": data.get("task_id"),
        }
        out.append(item)
    return out


def mbpp_get_predict(response):
    """MBPP is code generation; prediction is the code itself (no answer extraction)."""
    return response