# -*- coding: utf-8 -*-
"""goat_count 架构快照：SLOC / 模块边 / 环 / 无调用者符号 / 无用 import。只读。"""
import ast
import collections
import os
import re

ROOT = os.environ.get("ARCH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {".git", "__pycache__", "data", "runs", "out", ".venv", "venv"}
TYPING = {"Dict", "List", "Optional", "Sequence", "Tuple", "Iterable", "Iterator",
          "Any", "Callable", "Union", "Set", "Type", "Mapping"}

files = []
for d, dn, fns in os.walk(ROOT):
    dn[:] = [x for x in dn if x not in SKIP]
    files += [os.path.join(d, f) for f in fns if f.endswith(".py")]
files.sort()


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def modname(r):
    m = r[:-3].replace("/", ".")
    return m[:-9] if m.endswith(".__init__") else m


src = {rel(p): open(p, encoding="utf-8-sig").read() for p in files}
trees = {r: ast.parse(s, filename=r) for r, s in src.items()}


def sloc(s):
    return sum(1 for l in s.splitlines() if l.strip() and not l.strip().startswith("#"))


print("== 规模")
print(f"  python files {len(files)}  total sloc {sum(sloc(s) for s in src.values())}")
for r in sorted(src, key=lambda k: -sloc(src[k])):
    print(f"  {sloc(src[r]):5}  {r}")

# --- 模块依赖 ---
edges = set()
for r, t in trees.items():
    me = modname(r)
    pkg = me.rsplit(".", 1)[0] if "." in me else ""
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom):
            if n.level:
                base = pkg.split(".") if pkg else []
                if n.level > 1:
                    base = base[: len(base) - (n.level - 1)]
                tgt = ".".join(base + ([n.module] if n.module else []))
            else:
                tgt = n.module or ""
            if tgt:
                edges.add((me, tgt))
        elif isinstance(n, ast.Import):
            for a in n.names:
                edges.add((me, a.name))

internal = sorted({(a, b) for a, b in edges if b.startswith("goatcount") or b == "goatcount"})
fan_in = collections.Counter(b for _, b in internal)
fan_out = collections.Counter(a for a, _ in internal)

g = collections.defaultdict(set)
for a, b in internal:
    g[a].add(b)

cycles = []
def dfs(n, path):
    if n in path:
        k = tuple(sorted(set(path[path.index(n):])))
        if k not in cycles:
            cycles.append(k)
        return
    for x in g.get(n, ()):
        dfs(x, path + [n])
for n in list(g):
    dfs(n, [])

print("\n== 依赖")
print(f"  模块级边 {len(internal)}  环 {len(cycles)}")
print("  fan-in  ", ", ".join(f"{k}={v}" for k, v in fan_in.most_common(4)))
print("  fan-out ", ", ".join(f"{k}={v}" for k, v in fan_out.most_common(3)))

# --- 无调用者符号 ---
alltext = "\n".join(src.values())
dead = []
for r, t in trees.items():
    if not r.startswith("goatcount/"):
        continue
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("__"):
            if len(re.findall(r"\b" + re.escape(n.name) + r"\b", alltext)) == 1:
                dead.append(f"{n.name} ({r}:{n.lineno})")
print(f"\n== 无调用者符号 {len(dead)}")
for d in dead:
    print("  " + d)

# --- 无用 import ---
unused = []
for r, t in trees.items():
    future = "from __future__ import annotations" in src[r]
    body = src[r]
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom):
            names = [(a.asname or a.name, n.lineno) for a in n.names]
        elif isinstance(n, ast.Import):
            names = [((a.asname or a.name).split(".")[0], n.lineno) for a in n.names]
        else:
            continue
        if n.col_offset != 0:
            continue
        for nm, ln in names:
            if nm in ("annotations", "*"):
                continue
            uses = len(re.findall(r"\b" + re.escape(nm) + r"\b", body))
            if uses <= 1 or (future and nm in TYPING and uses <= 2):
                unused.append(f"{r}:{ln} {nm}")
print(f"\n== 无用 import {len(unused)}")
for u in unused:
    print("  " + u)
