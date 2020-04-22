import eqlib as eq
import numpy as np
from morphr.constraints.utility import evaluate_ref, evaluate_act, evaluate_act_2, normalized


class NormalCoupling(eq.Objective):
    def __init__(self, nodes_a, nodes_b, shape_functions_a, shape_functions_b, weight=1):
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

        angular_difference = a3_a - a3_b

        p = np.dot(angular_difference, angular_difference) * self.weight / 2

        g[:] = p.g
        h[:] = p.h
        return p.f
