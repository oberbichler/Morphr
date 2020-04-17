import eqlib as eq
import numpy as np
from morphr.constraints.utility import evaluate_ref, evaluate_act, evaluate_act_2, normalized


class RotationCoupling(eq.Objective):
    def __init__(self, nodes_a, nodes_b, shape_functions_a, shape_functions_b, axis=None, weight=1):
        eq.Objective.__init__(self)
        self.nodes_a = np.asarray(nodes_a, object)
        self.nodes_b = np.asarray(nodes_b, object)
        self.shape_functions_a = np.asarray(shape_functions_a, float)
        self.shape_functions_b = np.asarray(shape_functions_b, float)
        self.weight = float(weight)

        variables = []
        for node in nodes_a + nodes_b:
            variables += [node.x, node.y, node.z]
        self.variables = variables

        ref_a1_a = self.evaluate_ref_a(1)
        ref_a2_a = self.evaluate_ref_a(2)
        self.ref_a3_a = normalized(np.cross(ref_a1_a, ref_a2_a))

        ref_a1_b = self.evaluate_ref_b(1)
        ref_a2_b = self.evaluate_ref_b(2)
        self.ref_a3_b = normalized(np.cross(ref_a1_b, ref_a2_b))

        if axis is None:
            axis = np.cross(self.ref_a3_a, self.ref_a3_b)

        self.axis = normalized(np.asarray(axis, float))

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

    def evaluate_act_a(self, index):
        return evaluate_act(self.nodes_a, self.shape_functions_a[index])

    def evaluate_act_b(self, index):
        return evaluate_act(self.nodes_b, self.shape_functions_b[index])

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

        a3_a = normalized(np.cross(a1_a, a2_a))

        a1_b = self.evaluate_act_b_2(1)
        a2_b = self.evaluate_act_b_2(2)

        a3_b = normalized(np.cross(a1_b, a2_b))

        w_a = a3_a - self.ref_a3_a
        w_b = a3_b - self.ref_a3_b

        omega_a = np.cross(self.ref_a3_a, w_a)
        omega_b = np.cross(self.ref_a3_b, w_b)

        angle_a = np.arcsin(np.dot(omega_a, self.axis))
        angle_b = np.arcsin(np.dot(omega_b, self.axis))

        angular_difference = angle_a - angle_b

        p = angular_difference**2 * self.weight / 2

        g[:] = p.g
        h[:] = p.h
        return p.f
