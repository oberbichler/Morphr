import eqlib as eq
import numpy as np
import hyperjet as hj
from morphr.objectives.utility import evaluate_act, evaluate_act_geometry_hj_a, evaluate_act_geometry_hj_b, normalized


class IgaNormalDistanceAD(eq.Objective):
    def __init__(self, nodes_a, nodes_b):
        eq.Objective.__init__(self)
        self.nodes_a = np.asarray(nodes_a, object)
        self.nodes_b = np.asarray(nodes_b, object)

        variables = []
        for node in nodes_a:
            variables += [node.x, node.y, node.z]
        for node in nodes_b:
            variables += [node.x, node.y, node.z]
        self.variables = variables

        self.data = []

    def add(self, shape_functions_a, shape_functions_b, angle=0, weight=1):
        shape_functions_a = np.asarray(shape_functions_a, float)
        shape_functions_b = np.asarray(shape_functions_b, float)
        target = float(angle)  # 2 * np.sin(float(angle) / 2) if angle != 0 else 0.0
        weight = float(weight)

        self.data.append((shape_functions_a, shape_functions_b, target, weight))

    def compute(self, g, h):
        f = 0

        for shape_functions_a, shape_functions_b, target, weight in self.data:
            a1_a = evaluate_act(self.nodes_a, shape_functions_a[1])
            a2_a = evaluate_act(self.nodes_a, shape_functions_a[2])

            a1_b = evaluate_act(self.nodes_b, shape_functions_b[1])
            a2_b = evaluate_act(self.nodes_b, shape_functions_b[2])

            variables = hj.HyperJet.variables([*a1_a, *a2_a, *a1_b, *a2_b])

            a1_a = np.array(variables[0:3])
            a2_a = np.array(variables[3:6])
            a1_b = np.array(variables[6:9])
            a2_b = np.array(variables[9:12])

            a3_a = normalized(np.cross(a1_a, a2_a))
            a3_b = normalized(np.cross(a1_b, a2_b))

            delta = a3_a - a3_b
            p = np.dot(delta, delta) * weight / 2

            f += p.f

            a = len(shape_functions_a[0]) * 3
            b = len(shape_functions_b[0]) * 3

            for r in range(a):
                g[0 + r]  = p.g[0 + r % 3] * shape_functions_a[1, r // 3]
                g[0 + r] += p.g[3 + r % 3] * shape_functions_a[2, r // 3]

            for r in range(b):
                g[a + r]  = p.g[6 + r % 3] * shape_functions_b[1, r // 3]
                g[a + r] += p.g[9 + r % 3] * shape_functions_b[2, r // 3]

            for r in range(a):
                for s in range(a):
                    h[0 + r, 0 + s]  = p.h[0 + r % 3, 0 + s % 3] * shape_functions_a[1, r // 3] * shape_functions_a[1, s // 3]
                    h[0 + r, 0 + s] += p.h[3 + r % 3, 0 + s % 3] * shape_functions_a[2, r // 3] * shape_functions_a[1, s // 3]
                    h[0 + r, 0 + s] += p.h[0 + r % 3, 3 + s % 3] * shape_functions_a[1, r // 3] * shape_functions_a[2, s // 3]
                    h[0 + r, 0 + s] += p.h[3 + r % 3, 3 + s % 3] * shape_functions_a[2, r // 3] * shape_functions_a[2, s // 3]

                for s in range(b):
                    h[0 + r, a + s]  = p.h[0 + r % 3, 6 + s % 3] * shape_functions_a[1, r // 3] * shape_functions_b[1, s // 3]
                    h[0 + r, a + s] += p.h[3 + r % 3, 6 + s % 3] * shape_functions_a[2, r // 3] * shape_functions_b[1, s // 3]
                    h[0 + r, a + s] += p.h[0 + r % 3, 9 + s % 3] * shape_functions_a[1, r // 3] * shape_functions_b[2, s // 3]
                    h[0 + r, a + s] += p.h[3 + r % 3, 9 + s % 3] * shape_functions_a[2, r // 3] * shape_functions_b[2, s // 3]

            for r in range(b):
                for s in range(a):
                    h[a + r, 0 + s]  = p.h[6 + r % 3, 0 + s % 3] * shape_functions_b[1, r // 3] * shape_functions_a[1, s // 3]
                    h[a + r, 0 + s] += p.h[9 + r % 3, 0 + s % 3] * shape_functions_b[2, r // 3] * shape_functions_a[1, s // 3]
                    h[a + r, 0 + s] += p.h[6 + r % 3, 3 + s % 3] * shape_functions_b[1, r // 3] * shape_functions_a[2, s // 3]
                    h[a + r, 0 + s] += p.h[9 + r % 3, 3 + s % 3] * shape_functions_b[2, r // 3] * shape_functions_a[2, s // 3]

                for s in range(b):
                    h[a + r, a + s]  = p.h[6 + r % 3, 6 + s % 3] * shape_functions_b[1, r // 3] * shape_functions_b[1, s // 3]
                    h[a + r, a + s] += p.h[9 + r % 3, 6 + s % 3] * shape_functions_b[2, r // 3] * shape_functions_b[1, s // 3]
                    h[a + r, a + s] += p.h[6 + r % 3, 9 + s % 3] * shape_functions_b[1, r // 3] * shape_functions_b[2, s // 3]
                    h[a + r, a + s] += p.h[9 + r % 3, 9 + s % 3] * shape_functions_b[2, r // 3] * shape_functions_b[2, s // 3]

        return f
