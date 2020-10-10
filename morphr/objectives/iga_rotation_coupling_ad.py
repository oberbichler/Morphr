import eqlib as eq
import hyperjet as hj
import numpy as np
from morphr.objectives.utility import evaluate_ref, evaluate_act, evaluate_act_geometry_hj_a, evaluate_act_geometry_hj_b, normalized


class IgaRotationCouplingAD(eq.Objective):
    def __init__(self, nodes_a, nodes_b):
        eq.Objective.__init__(self)
        self.nodes_a = np.asarray(nodes_a, object)
        self.nodes_b = np.asarray(nodes_b, object)

        variables = []
        for node in nodes_a + nodes_b:
            variables += [node.x, node.y, node.z]
        self.variables = variables

        self.data = []

    def add(self, shape_functions_a, shape_functions_b, axis=None, weight=1):
        shape_functions_a = np.asarray(shape_functions_a, float)
        shape_functions_b = np.asarray(shape_functions_b, float)
        weight = float(weight)

        ref_a1_a = evaluate_ref(self.nodes_a, shape_functions_a[1])
        ref_a2_a = evaluate_ref(self.nodes_a, shape_functions_a[2])
        ref_a3_a = normalized(np.cross(ref_a1_a, ref_a2_a))

        ref_a1_b = evaluate_ref(self.nodes_b, shape_functions_b[1])
        ref_a2_b = evaluate_ref(self.nodes_b, shape_functions_b[2])
        ref_a3_b = normalized(np.cross(ref_a1_b, ref_a2_b))

        if axis is None:
            axis = np.cross(ref_a3_a, ref_a3_b)

        axis = normalized(np.asarray(axis, float))

        self.data.append((shape_functions_a, shape_functions_b, ref_a3_a, ref_a3_b, axis, weight))

    def compute(self, g, h):
        p = 0

        for shape_functions_a, shape_functions_b, ref_a3_a, ref_a3_b, axis, weight in self.data:
            a1_a = evaluate_act_geometry_hj_a(self.nodes_a, shape_functions_a[1], self.nb_variables)
            a2_a = evaluate_act_geometry_hj_a(self.nodes_a, shape_functions_a[2], self.nb_variables)

            a3_a = normalized(np.cross(a1_a, a2_a))

            a1_b = evaluate_act_geometry_hj_b(self.nodes_b, shape_functions_b[1], self.nb_variables)
            a2_b = evaluate_act_geometry_hj_b(self.nodes_b, shape_functions_b[2], self.nb_variables)

            a3_b = normalized(np.cross(a1_b, a2_b))

            w_a = a3_a - ref_a3_a
            w_b = a3_b - ref_a3_b

            omega_a = np.cross(ref_a3_a, w_a)
            omega_b = np.cross(ref_a3_b, w_b)

            angle_a = np.arcsin(np.dot(omega_a, axis))
            angle_b = np.arcsin(np.dot(omega_b, axis))

            angular_difference = angle_a - angle_b

            p += angular_difference**2 * weight

        return hj.explode(0.5 * p, g, h)
