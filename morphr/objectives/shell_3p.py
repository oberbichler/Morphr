import eqlib as eq
import numpy as np
from morphr.objectives.utility import evaluate_ref, evaluate_act, evaluate_act_geometry_hj


class Shell3P(eq.Objective):
    def __init__(self, nodes, thickness, youngs_modulus, poissons_ratio):
        eq.Objective.__init__(self)
        self.nodes = np.asarray(nodes, object)
        self.thickness = float(thickness)
        self.youngs_modulus = float(youngs_modulus)
        self.poissons_ratio = float(poissons_ratio)

        variables = []
        for node in nodes:
            variables += [node.x, node.y, node.z]
        self.variables = variables

        self.dm = np.array([
            [1.0, poissons_ratio, 0],
            [poissons_ratio, 1.0, 0],
            [0, 0, (1.0 - poissons_ratio) / 2.0],
        ]) * youngs_modulus * thickness / (1.0 - np.power(poissons_ratio, 2))

        self.db = np.array([
            [1.0, poissons_ratio, 0],
            [poissons_ratio, 1.0, 0],
            [0, 0, (1.0 - poissons_ratio) / 2.0],
        ]) * youngs_modulus * np.power(thickness, 3) / (12.0 * (1.0 - np.power(poissons_ratio, 2)))

        self.data = []

    def add(self, shape_functions, weight):
        shape_functions = np.asarray(shape_functions, float)
        weight = float(weight)

        A1 = evaluate_ref(self.nodes, shape_functions[1])
        A2 = evaluate_ref(self.nodes, shape_functions[2])

        A1_1 = evaluate_ref(self.nodes, shape_functions[3])
        A2_2 = evaluate_ref(self.nodes, shape_functions[4])
        A1_2 = evaluate_ref(self.nodes, shape_functions[5])

        A11 = np.dot(A1, A1)
        A22 = np.dot(A2, A2)
        A12 = np.dot(A1, A2)

        A3 = np.cross(A1, A2)
        dA = np.linalg.norm(A3)
        A3 = A3 / dA

        ref_a = np.array([A11, A22, A12])
        ref_b = np.dot([A1_1, A2_2, A1_2], A3)

        e1 = A1 / np.linalg.norm(A1)
        e2 = A2 - np.dot(A2, e1) * e1
        e2 /= np.linalg.norm(e2)

        det = A11 * A22 - A12 * A12

        g_ab_con = np.array([A22 / det, A11 / det, -A12 / det])

        g_con1 = g_ab_con[0] * A1 + g_ab_con[2] * A2
        g_con2 = g_ab_con[2] * A1 + g_ab_con[1] * A2

        eg11 = np.dot(e1, g_con1)
        eg12 = np.dot(e1, g_con2)
        eg21 = np.dot(e2, g_con1)
        eg22 = np.dot(e2, g_con2)

        tm = np.array([
            [eg11 * eg11, eg12 * eg12, 2 * eg11 * eg12],
            [eg21 * eg21, eg22 * eg22, 2 * eg21 * eg22],
            [2 * eg11 * eg21, 2 * eg12 * eg22, 2 * (eg11 * eg22 + eg12 * eg21)],
        ])

        self.data.append((shape_functions, ref_a, ref_b, tm, weight))

    def compute(self, g, h):
        p = 0

        for shape_functions, ref_a, ref_b, tm, weight in self.data:
            a1 = evaluate_act_geometry_hj(self.nodes, shape_functions[1])
            a2 = evaluate_act_geometry_hj(self.nodes, shape_functions[2])

            a1_1 = evaluate_act_geometry_hj(self.nodes, shape_functions[3])
            a1_2 = evaluate_act_geometry_hj(self.nodes, shape_functions[4])
            a2_2 = evaluate_act_geometry_hj(self.nodes, shape_functions[5])

            a3 = np.cross(a1, a2)
            a3 /= np.linalg.norm(a3)

            act_a = np.array([np.dot(a1, a1), np.dot(a2, a2), np.dot(a1, a2)])
            act_b = np.dot([a1_1, a1_2, a2_2], a3)

            eps = np.dot(tm, act_a - ref_a) / 2
            kap = np.dot(tm, ref_b - act_b)

            p += (np.dot(eps, np.dot(self.dm, eps)) + np.dot(kap, np.dot(self.db, kap))) * weight

        g[:] = p.g / 2
        h[:] = p.h / 2
        return p.f / 2
