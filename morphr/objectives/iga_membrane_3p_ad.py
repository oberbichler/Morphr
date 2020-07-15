import eqlib as eq
import hyperjet as hj
import numpy as np
from morphr.objectives.utility import evaluate_ref, evaluate_act_geometry_hj, normalized


class IgaMembrane3PAD(eq.Objective):
    def __init__(self, nodes, thickness, youngs_modulus, poissons_ratio, prestress=None):
        eq.Objective.__init__(self)
        self.nodes = np.asarray(nodes, object)
        self.thickness = float(thickness)
        self.youngs_modulus = float(youngs_modulus)
        self.poissons_ratio = float(poissons_ratio)
        self.prestress = np.zeros(3) if prestress is None else np.asarray(prestress)

        variables = []
        for node in nodes:
            variables += [node.x, node.y, node.z]
        self.variables = variables

        self.dm = np.array([
            [1.0, poissons_ratio, 0],
            [poissons_ratio, 1.0, 0],
            [0, 0, (1.0 - poissons_ratio) / 2.0],
        ]) * youngs_modulus * thickness / (1.0 - np.power(poissons_ratio, 2))

        self.data = []

    def add(self, shape_functions, weight):
        shape_functions = np.asarray(shape_functions, float)
        weight = float(weight)

        ref_a1 = evaluate_ref(self.nodes, shape_functions[1])
        ref_a2 = evaluate_ref(self.nodes, shape_functions[2])

        ref_a11 = np.dot(ref_a1, ref_a1)
        ref_a12 = np.dot(ref_a1, ref_a2)
        ref_a22 = np.dot(ref_a2, ref_a2)

        ref_a = np.array([ref_a11, ref_a22, ref_a12])

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

        self.data.append((shape_functions, ref_a, tm, weight))

    def compute(self, g, h):
        p = 0

        for shape_functions, ref_a, tm, weight in self.data:
            act_a1 = evaluate_act_geometry_hj(self.nodes, shape_functions[1])
            act_a2 = evaluate_act_geometry_hj(self.nodes, shape_functions[2])

            act_a = np.array([np.dot(act_a1, act_a1), np.dot(act_a2, act_a2), np.dot(act_a1, act_a2)])

            eps = np.dot(tm, act_a - ref_a) / 2

            p += np.dot(eps, np.add(np.dot(self.dm, eps), self.prestress)) * weight

        return hj.explode(0.5 * p, g, h)
