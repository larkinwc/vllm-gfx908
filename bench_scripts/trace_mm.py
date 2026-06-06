"""Find the parent CPU op that launches aten::mm, via a stack sweep (O(n log n))."""
import gzip, json, sys, collections

path = sys.argv[1]
with gzip.open(path) as f:
    data = json.load(f)
ev = data["traceEvents"]
cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation") and "dur" in e and "ts" in e]
by_tid = collections.defaultdict(list)
for e in cpu:
    by_tid[e["tid"]].append(e)

targets = {"aten::mm", "aten::matmul", "aten::bmm", "aten::addmm", "aten::linear"}
parents = {t: collections.Counter() for t in targets}

for tid, lst in by_tid.items():
    # sort by ts asc, then by dur desc so parents open before children
    lst.sort(key=lambda e: (e["ts"], -e["dur"]))
    stack = []  # (end_ts, name)
    for e in lst:
        ts, end = e["ts"], e["ts"] + e["dur"]
        while stack and stack[-1][0] <= ts:
            stack.pop()
        if e["name"] in targets:
            parents[e["name"]][stack[-1][1] if stack else "<root>"] += 1
        stack.append((end, e["name"]))

for t in targets:
    if parents[t]:
        print("== parents of %s ==" % t)
        for name, c in parents[t].most_common(8):
            print("  %6d  <- %s" % (c, name))
