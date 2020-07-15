import eqlib as eq
import numpy as np
from morphr.objectives.utility import evaluate_act_geometry_hj


class ReducedIgaShell(eq.Objective):
    def __init__(self, nodes, membrane_stiffness, bending_stiffness):
        eq.Objective.__init__(self)
        self.nodes = np.asarray(nodes, object)
        self.membrane_stiffness = float(membrane_stiffness)
        self.bending_stiffness = float(bending_stiffness)

        variables = []
        for node in nodes:
            variables += [node.x, node.y, node.z]
        self.variables = variables

        self.data = []

    def add(self, shape_functions, weight):
        self.data.append((shape_functions, weight))

    def compute(self, g, h):
        p = 0

        for shape_functions, weight in self.data:
            a1 = evaluate_act_geometry_hj(self.nodes, shape_functions[1])
            a2 = evaluate_act_geometry_hj(self.nodes, shape_functions[2])

            a1_1 = evaluate_act_geometry_hj(self.nodes, shape_functions[3])
            a1_2 = evaluate_act_geometry_hj(self.nodes, shape_functions[4])
            a2_2 = evaluate_act_geometry_hj(self.nodes, shape_functions[5])

            e_m = np.dot(a1, a1) + np.dot(a2, a2)
            e_b = np.dot(a1_1, a1_1) + np.dot(a1_2, a1_2) + np.dot(a2_2, a2_2)

            p += (e_m * self.membrane_stiffness + e_b * self.bending_stiffness) * weight

        return hj.explode(p, g, h)
