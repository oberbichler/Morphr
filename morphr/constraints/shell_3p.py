import eqlib as eq
import numpy as np
from morphr.constraints.utility import evaluate_ref, evaluate_act_2


class Shell3P(eq.Objective):
    def __init__(self, nodes, shape_functions, thickness, youngs_modulus, poissons_ratio, weight):
        eq.Objective.__init__(self)
        self.nodes = np.asarray(nodes, object)
        self.shape_functions = np.asarray(shape_functions, float)
        self.thickness = float(thickness)
        self.youngs_modulus = float(youngs_modulus)
        self.poissons_ratio = float(poissons_ratio)
        self.weight = float(weight)

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

        A1 = self.evaluate_ref(1)
        A2 = self.evaluate_ref(2)

        A1_1 = self.evaluate_ref(3)
        A2_2 = self.evaluate_ref(4)
        A1_2 = self.evaluate_ref(5)

        A11, A22, A12 = np.dot(A1, A1), np.dot(A2, A2), np.dot(A1, A2)

        A3 = np.cross(A1, A2)
        dA = np.linalg.norm(A3)
        A3 = A3 / dA

        self.ref_a = np.array([A11, A22, A12])
        self.ref_b = np.dot([A1_1, A2_2, A1_2], A3)

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

        self.tm = np.array([
            [eg11 * eg11, eg12 * eg12, 2 * eg11 * eg12],
            [eg21 * eg21, eg22 * eg22, 2 * eg21 * eg22],
            [2 * eg11 * eg21, 2 * eg12 * eg22, 2 * (eg11 * eg22 + eg12 * eg21)],
        ])

    def evaluate_ref(self, index):
        return evaluate_ref(self.nodes, self.shape_functions[index])

    def evaluate_act_2(self, index):
        nb_dofs = len(self.nodes) * 3
        return evaluate_act_2(self.nodes, self.shape_functions[index], nb_dofs, 0)

    def compute(self, g, h):
        a1 = self.evaluate_act_2(1)
        a2 = self.evaluate_act_2(2)

        a1_1 = self.evaluate_act_2(3)
        a1_2 = self.evaluate_act_2(4)
        a2_2 = self.evaluate_act_2(5)

        a3 = np.cross(a1, a2)
        a3 /= np.linalg.norm(a3)

        act_a = np.array([np.dot(a1, a1), np.dot(a2, a2), np.dot(a1, a2)])
        act_b = np.dot([a1_1, a1_2, a2_2], a3)

        eps = np.dot(self.tm, act_a - self.ref_a) / 2
        kap = np.dot(self.tm, self.ref_b - act_b)

        p = (np.dot(eps, np.dot(self.dm, eps)) + np.dot(kap, np.dot(self.db, kap))) * self.weight

        g[:] = p.g
        h[:] = p.h
        return p.f
