import eqlib as eq
import numpy as np
import hyperjet as hj


def normalized(v):
    return v / np.dot(v, v)**0.5


def evaluate_ref(nodes, shape_functions):
    xyz = np.zeros(3, float)

    for i, node in enumerate(nodes):
        xyz += shape_functions[i] * node.ref_location

    return xyz


def evaluate_act(nodes, shape_functions):
    xyz = np.zeros(3, float)

    for i, node in enumerate(nodes):
        xyz += shape_functions[i] * node.act_location

    return xyz


def evaluate_act_2(nodes, shape_functions, size, offset):
    xyz = evaluate_act(nodes, shape_functions)

    xyz = np.array(hj.HyperJet.constants(size, xyz))

    nb_dofs = len(shape_functions) * 3

    xyz[0].g[offset+0:offset+nb_dofs:3] = shape_functions
    xyz[1].g[offset+1:offset+nb_dofs:3] = shape_functions
    xyz[2].g[offset+2:offset+nb_dofs:3] = shape_functions

    return xyz


class PointOnSurfaceSupport(eq.Objective):
    def __init__(self, nodes, shape_functions, target, weight):
        eq.Objective.__init__(self)
        self.nodes = nodes
        self.shape_functions = shape_functions
        self.target = target
        self.weight = weight
        variables = []
        for node in nodes:
            variables += [node.x, node.y, node.z]
        self.variables = variables

    @property
    def act_location(self):
        return evaluate_act(self.nodes, self.shape_functions[0])

    def evaluate_act_2(self, index):
        nb_dofs = len(self.nodes) * 3
        return evaluate_act_2(self.nodes, self.shape_functions[index], nb_dofs, 0)

    def compute(self, g, h):
        x = self.evaluate_act_2(0)

        delta = self.target - x

        p = np.dot(delta, delta) * self.weight / 2

        g[:] = p.g
        h[:] = p.h
        return p.f


class EdgeRotationCoupling(eq.Objective):
    def __init__(self, nodes_a, nodes_b, shape_functions_a, shape_functions_b, t2_edge, weight):
        eq.Objective.__init__(self)
        self.nodes_a = nodes_a
        self.nodes_b = nodes_b
        self.shape_functions_a = shape_functions_a
        self.shape_functions_b = shape_functions_b
        self.weight = weight
        variables = []
        for node in nodes_a + nodes_b:
            variables += [node.x, node.y, node.z]
        self.variables = variables

        ref_a1_a = self.evaluate_ref_a(1)
        ref_a2_a = self.evaluate_ref_a(2)
        self.ref_a3_a = np.cross(ref_a1_a, ref_a2_a)
        self.ref_a3_a /= np.linalg.norm(self.ref_a3_a)

        ref_a1_b = self.evaluate_ref_b(1)
        ref_a2_b = self.evaluate_ref_b(2)
        self.ref_a3_b = np.cross(ref_a1_b, ref_a2_b)
        self.ref_a3_b /= np.linalg.norm(self.ref_a3_b)

        self.t2_edge = t2_edge

    @property
    def act_a(self):
        return evaluate_act(self.nodes_a, self.shape_functions_a[0])

    @property
    def act_b(self):
        return evaluate_act(self.nodes_b, self.shape_functions_b[0])

    def evaluate_ref_a(self, index):
        return evaluate_ref(self.nodes_a, self.shape_functions_a[index])

    def evaluate_ref_b(self, index):
        return evaluate_ref(self.nodes_b, self.shape_functions_b[index])

    def evaluate_act_a_2(self, index):
        nb_dofs_a = len(self.nodes_a) * 3
        nb_dofs_b = len(self.nodes_b) * 3
        return evaluate_act_2(self.nodes_a, self.shape_functions_a[index], nb_dofs_a + nb_dofs_b, 0)

    def evaluate_act_b_2(self, index):
        nb_dofs_a = len(self.nodes_a) * 3
        nb_dofs_b = len(self.nodes_b) * 3
        return evaluate_act_2(self.nodes_b, self.shape_functions_b[index], nb_dofs_a + nb_dofs_b, nb_dofs_a)

    def compute(self, g, h):
        a1_a = self.evaluate_act_a_2(1)
        a2_a = self.evaluate_act_a_2(2)

        a3_a = np.cross(a1_a, a2_a)
        a3_a /= np.linalg.norm(a3_a)

        a1_b = self.evaluate_act_b_2(1)
        a2_b = self.evaluate_act_b_2(2)

        a3_b = np.cross(a1_b, a2_b)
        a3_b /= np.linalg.norm(a3_b)

        w_a = a3_a - self.ref_a3_a
        w_b = a3_b - self.ref_a3_b

        omega_a = np.cross(self.ref_a3_a, w_a)
        omega_b = np.cross(self.ref_a3_b, w_b)

        angle_a = np.arcsin(np.dot(omega_a, self.t2_edge))
        angle_b = np.arcsin(np.dot(omega_b, self.t2_edge))

        angular_difference = angle_a - angle_b

        p = angular_difference**2 * self.weight / 2

        g[:] = p.g
        h[:] = p.h
        h.fill(0)
        return p.f


class Shell3P3D(eq.Objective):
    def __init__(self, nodes, shape_functions, thickness, youngs_modulus, poissons_ratio, weight):
        eq.Objective.__init__(self)
        self.nodes = nodes
        self.shape_functions = shape_functions
        self.thickness = thickness
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio
        self.weight = weight
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
        act_b = np.dot([a1_1, a2_2, a1_2], a3)

        eps = np.dot(self.tm, act_a - self.ref_a) / 2
        kap = np.dot(self.tm, self.ref_b - act_b)

        p = (np.dot(eps, np.dot(self.dm, eps)) + np.dot(kap, np.dot(self.db, kap))) * self.weight

        g[:] = p.g
        h[:] = p.h
        return p.f
