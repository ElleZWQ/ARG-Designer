import re


def mbpp_data_process(dataset):
    """Convert raw MBPP records into {task, test} rows.

    Each MBPP record has: text (natural-language task), test_list (list of
    assert statements). We build a single test string by joining them, matching
    how HumanEval stores its single `test` field, so PyExecutor can run it.
    """
    out = []
    for data in dataset:
        test_list = data.get('test_list') or []
        test = "\n".join(test_list)
        if data.get('test_setup_code'):
            test = data['test_setup_code'] + "\n" + test
        item = {
            "task": data["text"],
            "test": test,
            "task_id": data.get("task_id"),
        }
        out.append(item)
    return out


def mbpp_get_predict(response):
    """MBPP is code generation; prediction is the code itself (no answer extraction)."""
    return response