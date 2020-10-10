import eqlib as eq
import hyperjet as hj
import numpy as np
from morphr.objectives.utility import evaluate_ref, evaluate_act, evaluate_act_geometry_hj


class IgaPointLocationAD(eq.Objective):
    def __init__(self, nodes):
        eq.Objective.__init__(self)
        self.nodes = np.asarray(nodes, object)

        variables = []
        for node in nodes:
            variables += [node.x, node.y, node.z]
        self.variables = variables

        self.data = []

    def add(self, shape_functions, target, weight):
        shape_functions = np.asarray(shape_functions, float)
        target = self.ref_location if target is None else np.asarray(target, float)
        weight = float(weight)

        self.data.append((shape_functions, target, weight))

    @property
    def ref_location(self):
        return evaluate_ref(self.nodes, self.shape_functions[0])

    @property
    def act_location(self):
        return evaluate_act(self.nodes, self.shape_functions[0])

    def compute(self, g, h):
        p = 0

        for shape_functions, target, weight in self.data:
            act_x = evaluate_act_geometry_hj(self.nodes, shape_functions[0])

            delta = target - act_x

            p += weight * np.dot(delta, delta)

        return hj.explode(0.5 * p, g, h)
