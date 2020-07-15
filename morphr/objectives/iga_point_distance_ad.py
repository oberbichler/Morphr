import eqlib as eq
import hyperjet as hj
import numpy as np
from morphr.objectives.utility import evaluate_act, evaluate_act_geometry_hj_a, evaluate_act_geometry_hj_b


class IgaPointDistanceAD(eq.Objective):
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

    def add(self, shape_functions_a, shape_functions_b, weight):
        shape_functions_a = np.asarray(shape_functions_a, float)
        shape_functions_b = np.asarray(shape_functions_b, float)
        weight = float(weight)

        self.data.append((shape_functions_a, shape_functions_b, weight))

        return len(self.data) - 1

    def compute(self, g, h):
        p = 0

        for shape_functions_a, shape_functions_b, weight in self.data:
            x_a = evaluate_act_geometry_hj_a(self.nodes_a, shape_functions_a[0], self.nb_variables)
            x_b = evaluate_act_geometry_hj_b(self.nodes_b, shape_functions_b[0], self.nb_variables)

            delta = x_b - x_a

            p += weight * np.dot(delta, delta)

        return hj.explode(0.5 * p, g, h)
