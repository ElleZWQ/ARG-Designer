import re

try:
    from sympy import simplify
    from sympy.parsing.sympy_parser import parse_expr
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False


def _strip_string(s):
    s = str(s).strip()
    # LaTeX wrappers and spacing
    for tok in ['\\left', '\\right', '\\!', '\\,', '\\;', '\\quad']:
        s = s.replace(tok, '')
    return s.strip()


def _last_boxed(s):
    i = s.rfind('\\boxed')
    if i < 0:
        return None
    j = s.find('{', i)
    if j < 0:
        return None
    stack, out = 0, ''
    for c in s[j:]:
        if c == '{':
            stack += 1
            if stack > 1:
                out += c
        elif c == '}':
            stack -= 1
            if stack == 0:
                break
            out += c
        else:
            out += c
    return out.strip()


def math_data_process(dataset):
    """MATH-500 records already carry `problem` and a standalone `answer`.

    We build {task, answer} rows. `answer` is a LaTeX string (e.g.
    "\\left(3, \\frac{\\pi}{2}\\right)"); the model is asked to produce the
    answer, and correctness is judged via symbolic equivalence.
    """
    out = []
    for data in dataset:
        out.append({
            "task": data["problem"],
            "answer": str(data.get("answer", "")).strip(),
            "task_id": data.get("unique_id", data.get("problem", "")),
        })
    return out


def _latex_to_expr(s):
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\!', '').replace('\\,', '').replace('\\;', '')
    s = re.sub(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
    s = re.sub(r'\\d?frac(\d)(\d)', r'((\1)/(\2))', s)
    s = s.replace('\\pi', 'pi').replace('\\cdot', '*').replace('\\times', '*')
    s = s.replace('^', '**').replace('\\sqrt', 'sqrt')
    s = s.replace('{', '(').replace('}', ')')
    s = s.replace('%', '').replace('\\$', '').replace('$', '').replace('\\', '')
    return s.strip()


def math_get_predict(response):
    """Extract the model's final answer, preferring \\boxed{...}."""
    if not response:
        return ''
    b = _last_boxed(response)
    if b:
        return b
    m = re.search(r'answer is[:\s\$]*(.+?)[\.\n\$]', response, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return response.strip()


def math_check_correctness(predict, true_answer):
    """True if predicted answer is symbolically equivalent to the gold answer."""
    if not predict:
        return False
    p, g = _strip_string(predict), _strip_string(true_answer)
    if p == g:
        return True
    try:
        if abs(float(p) - float(g)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    if _HAS_SYMPY:
        try:
            pe = parse_expr(_latex_to_expr(p))
            ge = parse_expr(_latex_to_expr(g))
            if simplify(pe - ge) == 0:
                return True
        except Exception:
            pass
    return False