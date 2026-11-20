"""Generate owned scalar reverse-mode CUDA for the unchanged stable equations.

Prints a header; no runtime dependency or finite differences. The expression
graph preserves forward operation order and fixed-camera10-input derivatives.
Camera-gradient mode remains the explicitly implemented directional CUDA path.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import sys


class Graph:
    def __init__(self):
        self.nodes = []
        self.cache = {}

    def node(self, op, *args):
        key = (op, *args)
        if key not in self.cache:
            index = len(self.nodes)
            active = op == "variable" or (
                op not in ("constant", "fixed") and any(self.nodes[x][2] for x in args)
            )
            self.nodes.append((op, args, active))
            self.cache[key] = E(self, index)
        return self.cache[key]

    def value(self, x):
        return x if isinstance(x, E) else self.node("constant", float(x))


class E:
    def __init__(self, graph, index):
        self.g, self.i = graph, index

    def binary(self, op, other):
        return self.g.node(op, self.i, self.g.value(other).i)

    def __add__(self, other):
        return self.binary("add", other)

    def __sub__(self, other):
        return self.binary("sub", other)

    def __mul__(self, other):
        return self.binary("mul", other)

    def __truediv__(self, other):
        return self.binary("div", other)

    def __neg__(self):
        return self.g.node("neg", self.i)

    def __radd__(self, other):
        return self.g.value(other) + self

    def __rsub__(self, other):
        return self.g.value(other) - self

    def __rmul__(self, other):
        return self.g.value(other) * self


def equations(g):
    m = [g.node("variable", f"means[{i}]") for i in range(3)]
    q = [g.node("variable", f"quats[{i}]") for i in range(4)]
    s = [g.node("variable", f"scales[{i}]") for i in range(3)]
    leaves = m + q + s

    def fixed(text):
        return g.node("fixed", text)

    def root(a):
        return g.node("sqrt", a.i)

    def low(a, b):
        return a.binary("max", b)

    def up(a, b):
        return a.binary("min", b)

    norm = low(root(sum(x * x for x in q)), 1e-12)
    q = [x / norm for x in q]
    R = [[fixed(f"T[{4 * i + j}]") for j in range(3)] for i in range(3)]
    local = [
        sum((R[i][j] * m[j] for j in range(3)), fixed(f"T[{4 * i + 3}]"))
        for i in range(3)
    ]
    w, x, y, zq = q
    A = [
        [1 - 2 * (y * y + zq * zq), 2 * (x * y - w * zq), 2 * (x * zq + w * y)],
        [2 * (x * y + w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w * x)],
        [2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)],
    ]
    Cw = [
        [sum((A[i][k] * (s[k] * s[k])) * A[j][k] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    tmp = [
        [sum(R[i][k] * Cw[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]
    C = [
        [sum(tmp[i][k] * R[j][k] for k in range(3)) for j in range(3)] for i in range(3)
    ]
    jitter = low(
        fixed("relative") * low(C[0][0] + C[1][1] + C[2][2], fixed("absolute")),
        fixed("absolute"),
    )
    S = [
        [(C[i][j] + C[j][i]) * 0.5 + (jitter if i == j else 0) for j in range(3)]
        for i in range(3)
    ]
    z = local[2]
    fx, fy, cx, cy = [fixed(f"K[{i}]") for i in [0, 4, 2, 5]]
    u = up(
        low(local[0] / z, -cx / fx - fixed(".15*W") / fx),
        (fixed("W") - cx) / fx + fixed(".15*W") / fx,
    )
    v = up(
        low(local[1] / z, -cy / fy - fixed(".15*H") / fy),
        (fixed("H") - cy) / fy + fixed(".15*H") / fy,
    )
    ray = [u, v, g.value(1)]
    L = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            a = S[i][j]
            for k in range(j):
                a = a - L[i][k] * L[j][k]
            L[i][j] = root(a) if i == j else a / L[j][j]
    b, h = [None] * 3, [None] * 3
    for i in range(3):
        a = ray[i]
        for k in range(i):
            a = a - L[i][k] * b[k]
        b[i] = a / L[i][i]
    for i in range(2, -1, -1):
        a = b[i]
        for k in range(i + 1, 3):
            a = a - L[k][i] * h[k]
        h[i] = a / L[i][i]
    den = h[0] * u + h[1] * v + h[2]
    h = [x / den for x in h]
    p0 = (v * v + 1) * h[0] - u * v * h[1] - u * h[2]
    p1 = -u * v * h[0] + (u * u + 1) * h[1] - v * h[2]
    tx, ty = z * u, z * v
    dist = root(tx * tx + ty * ty + z * z)
    factor = dist / (u * u + v * v + 1)
    rn0, rn1 = -p0 * factor, -p1 * factor
    n = [
        rn0 / z - tx / dist,
        rn1 / z - ty / dist,
        -tx / (z * z) * rn0 - ty / (z * z) * rn1 - z / dist,
    ]
    nn = low(root(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]), 1e-12)
    return leaves, [
        root(local[0] * local[0] + local[1] * local[1] + local[2] * local[2]),
        p0 * factor / fx,
        p1 * factor / fy,
        *[x / nn for x in n],
    ]


def generate():
    g = Graph()
    leaves, outputs = equations(g)
    code = [
        "// Generated by tools/generate_ray_geometry_reverse.py; do not hand edit.",
        "// SPDX-License-Identifier: Apache-2.0",
        "#pragma once",
        "__device__ void reverse_fixed(const float* means,const float* quats,const float* scales,",
        " const float* T,const float* K,int W,int H,double relative,double absolute,",
        " const double* incoming,float* gm,float* gq,float* gs) {",
    ]
    for i, (op, args, active) in enumerate(g.nodes):
        names = (
            [f"v{x}" for x in args]
            if op not in ("constant", "variable", "fixed")
            else []
        )
        if op == "constant":
            expr = repr(args[0])
        elif op in ("variable", "fixed"):
            expr = f"double({args[0]})"
        elif op in ("add", "sub", "mul", "div"):
            expr = (
                f"{names[0]}" + dict(add="+", sub="-", mul="*", div="/")[op] + names[1]
            )
        elif op == "sqrt":
            expr = f"sqrt({names[0]})"
        elif op == "neg":
            expr = "-" + names[0]
        else:
            expr = (
                f"({names[0]}"
                + (">=" if op == "max" else "<=")
                + f"{names[1]}?{names[0]}:{names[1]})"
            )
        code.append(f"  double v{i}={expr};")
        if active:
            code.append(f"  double b{i}=0.;")
    for k, x in enumerate(outputs):
        code.append(f"  b{x.i}+=incoming[{k}];")
    for i in range(len(g.nodes) - 1, -1, -1):
        op, args, active = g.nodes[i]
        if not active or op == "variable":
            continue
        updates = []

        def push(j, expr):
            if g.nodes[args[j]][2]:
                updates.append(f"b{args[j]}+={expr};")

        if op == "add":
            push(0, f"b{i}")
            push(1, f"b{i}")
        elif op == "sub":
            push(0, f"b{i}")
            push(1, f"-b{i}")
        elif op == "neg":
            push(0, f"-b{i}")
        elif op == "mul":
            push(0, f"b{i}*v{args[1]}")
            push(1, f"b{i}*v{args[0]}")
        elif op == "div":
            push(0, f"b{i}/v{args[1]}")
            push(1, f"-b{i}*v{args[0]}/(v{args[1]}*v{args[1]})")
        elif op == "sqrt":
            push(0, f"b{i}/(2.*v{i})")
        elif op in ("max", "min"):
            condition = f"v{args[0]}" + (">=" if op == "max" else "<=") + f"v{args[1]}"
            push(0, f"({condition}?b{i}:0.)")
            push(1, f"({condition}?0.:b{i})")
        code.append(f"  if(b{i}!=0.) {{" + "".join(updates) + "}")
    for j, x in enumerate(leaves):
        target, index = (
            ("gm", j) if j < 3 else ("gq", j - 3) if j < 7 else ("gs", j - 7)
        )
        code.append(f"  {target}[{index}]=float(b{x.i});")
    code += ["}", "// END GENERATED REVERSE"]
    return "\n".join(code) + "\n", len(g.nodes)


if __name__ == "__main__":
    source, count = generate()
    print(
        count if "--stats" in sys.argv else source,
        end="\n" if "--stats" in sys.argv else "",
    )
