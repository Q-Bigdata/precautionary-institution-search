
"""Validation, compilation and description length of candidate allocation programs."""
import ast, builtins
import numpy as np

SIG = ["pool", "prev_offer", "prev_contrib", "t", "mem", "cap", "growth"]
BANNED_NAMES = {"eval", "exec", "open", "compile", "globals", "locals", "getattr", "setattr", "delattr",
                "vars", "input", "breakpoint", "__import__", "exit", "quit", "help", "memoryview", "type", "object",
                "super", "classmethod", "staticmethod", "property"}
BANNED_ATTRS = {"random", "load", "save", "savetxt", "loadtxt", "fromfile", "tofile", "memmap", "ctypeslib", "lib",
                "testing", "distutils", "f2py", "os", "sys"}
SAFE_BUILTINS = {k: getattr(builtins, k) for k in ["abs", "min", "max", "range", "len", "float", "int", "sum",
                 "enumerate", "zip", "isinstance", "dict", "list", "tuple", "bool", "round", "pow", "any", "all",
                 "sorted", "reversed", "ValueError", "Exception"]}
# __import__ is needed by numpy/warnings internals; programs cannot reach it (Import nodes and dunder names are rejected)
SAFE_BUILTINS["__import__"] = builtins.__import__

class Invalid(Exception):
    pass

def validate(code, max_chars=6000):
    if len(code) > max_chars:
        raise Invalid("too long")
    tree = ast.parse(code)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(f.name == "allocate" for f in fns):
        raise Invalid("no allocate()")
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.Assign, ast.Expr)):
            raise Invalid(f"top-level {type(n).__name__} not allowed")
        if isinstance(n, ast.Expr) and not isinstance(getattr(n, "value", None), ast.Constant):
            raise Invalid("top-level expression")
    f = [f for f in fns if f.name == "allocate"][0]
    args = [a.arg for a in f.args.args]
    if args != SIG:
        raise Invalid(f"signature {args}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.While, ast.ClassDef,
                             ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom, ast.With, ast.Raise)):
            raise Invalid(f"{type(node).__name__} not allowed")
        if isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise Invalid(f"name {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BANNED_ATTRS):
            raise Invalid(f"attr {node.attr}")
    return tree

def compile_rule(code):
    validate(code)
    ns = {"np": np, "__builtins__": SAFE_BUILTINS}
    exec(compile(code, "<program>", "exec"), ns)
    return ns["allocate"]

def description_length(code):
    """AST node count of the module after removing docstrings (constants count as one node)."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Module)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return sum(1 for n in ast.walk(tree) if not isinstance(n, (ast.Load, ast.Store, ast.Del)))

def logical_lines(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Module)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.stmt)) - 1

def smoke(rule, G=7, P=4, rng=None):
    rng = rng or np.random.default_rng(0)
    mem = {}
    for t in range(1, 6):
        pool = rng.uniform(0, 200, G); pool[0] = 0.0; pool[1] = 200.0
        po = rng.uniform(0, 50, (G, P)); pc = po * rng.uniform(0, 1, (G, P)); pc[2] = 0.0
        out = np.asarray(rule(pool, po, pc, t, mem, 200.0, 0.4), dtype=float)
        out = np.broadcast_to(out, (G, P))
        if not np.all(np.isfinite(out)):
            raise Invalid("non-finite offers")
    return True


def shares_wrapper(rule):
    """Share interface: the program returns non-negative weights and the whole reservoir is offered
    in proportion to them (equal split if all weights are zero)."""
    def allocate(pool, prev_offer, prev_contrib, t, mem, cap, growth):
        w = np.maximum(np.nan_to_num(np.asarray(rule(pool, prev_offer, prev_contrib, t, mem, cap, growth), float),
                                     nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        w = np.broadcast_to(w, prev_contrib.shape)
        s = w.sum(1, keepdims=True)
        return np.where(s > 0, pool[:, None] * w / np.maximum(s, 1e-300), pool[:, None] / prev_contrib.shape[1])
    return allocate

def compile_item(item):
    rule = compile_rule(item["code"])
    return shares_wrapper(rule) if item.get("wrap") == "shares" else rule
