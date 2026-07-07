"""Compare two corr_gen.py dumps: leading token-match length + logprob deltas."""
import json, sys
a = json.load(open(sys.argv[1]))  # gemv
b = json.load(open(sys.argv[2]))  # baseline
sa, sb = a["seq"], b["seq"]
n = min(len(sa), len(sb))
match = 0
for i in range(n):
    if sa[i][0] == sb[i][0]:
        match += 1
    else:
        break
# logprob delta over the matched prefix
deltas = [abs(sa[i][1] - sb[i][1]) for i in range(match)]
maxd = max(deltas) if deltas else 0.0
meand = sum(deltas) / len(deltas) if deltas else 0.0
print("tokens: gemv=%d baseline=%d compared=%d" % (len(sa), len(sb), n))
print("leading exact token match: %d / %d" % (match, n))
print("chosen-token logprob delta over matched prefix: max=%.5f mean=%.5f" % (maxd, meand))
if match < n:
    print("first divergence at token %d: gemv id=%d (lp %.4f) vs base id=%d (lp %.4f)"
          % (match, sa[match][0], sa[match][1], sb[match][0], sb[match][1]))
print("GEMV text head:", repr(a["text"][:120]))
print("BASE text head:", repr(b["text"][:120]))
verdict = "PASS" if (match == n or maxd < 0.05) else "REVIEW"
print("VERDICT:", verdict)
