#!/usr/bin/env python3
"""Speculative-decoding feasibility: expert-union growth over consecutive tokens + prompt-lookup draft acceptance."""
import sys
def load(p):
    P, G, R = [], [], []
    for line in open(p):
        parts = line.split()
        if parts[0] == 'P': P.append(int(parts[1]))
        else:
            G.append(int(parts[1]))
            R.append([set(int(e) for e in l.split(',')) for l in parts[2].split(';')])
    return P, G, R
GU, DN = 2.986, 1.66  # MB per expert (gate_up, avg down)
SHARED = 1260.0       # MB per pass: attention/ctrl/router weights + lm_head
for p in sys.argv[1:]:
    P, G, R = load(p)
    n = len(G)
    print(f"== {p}: prompt {len(P)} tok, generated {n}")
    for B in (2, 3, 4, 6, 8):
        us = []
        for i in range(0, n - B + 1):
            us.append(sum(len(set().union(*[R[i + j][l] for j in range(B)])) for l in range(48)) / 48)
        u = sum(us) / len(us)
        cost = SHARED + 48 * u * (GU + DN)
        single = SHARED + 48 * 8 * (GU + DN)
        print(f"  B={B}: mean experts/layer {u:5.1f} (random {128*(1-(1-8/128)**B):5.1f}) -> pass cost {cost/single:4.2f}x a single-token step")
    # prompt-lookup drafting: longest suffix match (n=1..4) against all earlier tokens, propose up to K
    ctx = list(P)
    for K in (3, 5, 8):
        acc_tokens, passes, i = 0, 0, 0
        ctx = list(P)
        while i < n:
            draft = []
            for ng in (4, 3, 2):
                if len(ctx) < ng: continue
                key = ctx[-ng:]
                for j in range(len(ctx) - ng - 1, -1, -1):
                    if ctx[j:j + ng] == key:
                        draft = ctx[j + ng:j + ng + K]
                        break
                if draft: break
            a = 0
            while a < len(draft) and i + a < n and draft[a] == G[i + a]: a += 1
            step = min(a + 1, n - i)          # accepted drafts + the verified token
            ctx += G[i:i + step]
            i += step
            acc_tokens += step
            passes += 1
        print(f"  prompt-lookup K={K}: {acc_tokens/passes:4.2f} tokens per verification pass")
