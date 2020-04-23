import eqlib as eq
import numpy as np
from morphr.constraints.utility import evaluate_act, evaluate_act_geometry_hj_a, evaluate_act_geometry_hj_b, normalized


class NormalDistance(eq.Objective):
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

    def compute(self, g, h):
        p = 0

        for shape_functions_a, shape_functions_b, weight in self.data:
            a1_a = evaluate_act_geometry_hj_a(self.nodes_a, shape_functions_a[1], self.nb_variables)
            a2_a = evaluate_act_geometry_hj_a(self.nodes_a, shape_functions_a[2], self.nb_variables)

            a3_a = normalized(np.cross(a1_a, a2_a))

            a1_b = evaluate_act_geometry_hj_b(self.nodes_b, shape_functions_b[1], self.nb_variables)
            a2_b = evaluate_act_geometry_hj_b(self.nodes_b, shape_functions_b[2], self.nb_variables)

            a3_b = normalized(np.cross(a1_b, a2_b))

            delta = a3_a - a3_b

            p += np.dot(delta, delta) * weight / 2

        g[:] = p.g
        h[:] = p.h
        return p.f
