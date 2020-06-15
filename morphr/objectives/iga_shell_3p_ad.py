import eqlib as eq
import numpy as np
from morphr.objectives.utility import evaluate_ref, evaluate_act, evaluate_act_geometry_hj, normalized


class IgaShell3PAD(eq.Objective):
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

        ref_a1 = evaluate_ref(self.nodes, shape_functions[1])
        ref_a2 = evaluate_ref(self.nodes, shape_functions[2])

        ref_a1_1 = evaluate_ref(self.nodes, shape_functions[3])
        ref_a1_2 = evaluate_ref(self.nodes, shape_functions[4])
        ref_a2_2 = evaluate_ref(self.nodes, shape_functions[5])

        ref_a11 = np.dot(ref_a1, ref_a1)
        ref_a12 = np.dot(ref_a1, ref_a2)
        ref_a22 = np.dot(ref_a2, ref_a2)

        ref_a3 = normalized(np.cross(ref_a1, ref_a2))

        ref_a = np.array([ref_a11, ref_a22, ref_a12])
        ref_b = np.dot([ref_a1_1, ref_a1_2, ref_a2_2], ref_a3)

        e1 = ref_a1 / np.linalg.norm(ref_a1)
        e2 = ref_a2 - np.dot(ref_a2, e1) * e1
        e2 /= np.linalg.norm(e2)

        det = ref_a11 * ref_a22 - ref_a12 * ref_a12

        g_ab_con = np.array([ref_a22 / det, ref_a11 / det, -ref_a12 / det])

        g_con1 = g_ab_con[0] * ref_a1 + g_ab_con[2] * ref_a2
        g_con2 = g_ab_con[2] * ref_a1 + g_ab_con[1] * ref_a2

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
            act_a1 = evaluate_act_geometry_hj(self.nodes, shape_functions[1])
            act_a2 = evaluate_act_geometry_hj(self.nodes, shape_functions[2])

            act_a1_1 = evaluate_act_geometry_hj(self.nodes, shape_functions[3])
            act_a1_2 = evaluate_act_geometry_hj(self.nodes, shape_functions[4])
            act_a2_2 = evaluate_act_geometry_hj(self.nodes, shape_functions[5])

            act_a3 = np.cross(act_a1, act_a2)
            act_a3 /= np.linalg.norm(act_a3)

            act_a = np.array([np.dot(act_a1, act_a1), np.dot(act_a2, act_a2), np.dot(act_a1, act_a2)])
            act_b = np.dot([act_a1_1, act_a1_2, act_a2_2], act_a3)

            eps = np.dot(tm, act_a - ref_a) / 2
            kap = np.dot(tm, act_b - ref_b)

            p += (np.dot(eps, np.dot(self.dm, eps)) + np.dot(kap, np.dot(self.db, kap))) * weight

        g[:] = p.g / 2
        h[:] = p.h / 2
        return p.f / 2
