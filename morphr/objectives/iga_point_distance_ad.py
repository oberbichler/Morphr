import eqlib as eq
import numpy as np
from morphr.objectives.utility import evaluate_act, evaluate_act_geometry_hj_a, evaluate_act_geometry_hj_b


class IgaPointDistanceAD(eq.Objective):
    def __init__(self, nodes_a, nodes_b):
        eq.Objective.__init__(self)
        self.nodes_a = np.asarray(nodes_a, object)
        self.nodes_b = np.asarray(nodes_b, object)

        variables = []
        for node in nodes_a + nodes_b:
            variables += [node.x, node.y, node.z]
        self.variables = variables

        self.data = []

    def add(self, shape_functions_a, shape_functions_b, weight):
        shape_functions_a = np.asarray(shape_functions_a, float)
        shape_functions_b = np.asarray(shape_functions_b, float)
        weight = float(weight)

        self.data.append((shape_functions_a, shape_functions_b, weight))

        return len(self.data) - 1

    @property
    def act_a(self):
        return evaluate_act(self.nodes_a, self.shape_functions_a[0])

    @property
    def act_b(self):
        return evaluate_act(self.nodes_b, self.shape_functions_b[0])

    def evaluate_act_a_2(self, index):
        nb_dofs_a = len(self.nodes_a) * 3
        nb_dofs_b = len(self.nodes_b) * 3
        return evaluate_act_2(self.nodes_a, self.shape_functions_a[index], nb_dofs_a + nb_dofs_b, 0)

    def evaluate_act_b_2(self, index):
        nb_dofs_a = len(self.nodes_a) * 3
        nb_dofs_b = len(self.nodes_b) * 3
        return evaluate_act_2(self.nodes_b, self.shape_functions_b[index], nb_dofs_a + nb_dofs_b, nb_dofs_a)

    def compute(self, g, h):
        p = 0

        for shape_functions_a, shape_functions_b, weight in self.data:
            x_a = evaluate_act_geometry_hj_a(self.nodes_a, shape_functions_a[0], self.nb_variables)
            x_b = evaluate_act_geometry_hj_b(self.nodes_b, shape_functions_b[0], self.nb_variables)

            delta = x_b - x_a

            p += np.dot(delta, delta) * weight

        g[:] = p.g / 2
        h[:] = p.h / 2
        return p.f / 2
