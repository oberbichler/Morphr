import eqlib as eq
import numpy as np
from morphr.constraints.utility import evaluate_ref, evaluate_act, evaluate_act_2


class PointSupport(eq.Objective):
    def __init__(self, nodes, shape_functions, target, weight):
        eq.Objective.__init__(self)
        self.nodes = np.asarray(nodes, object)
        self.shape_functions = np.asarray(shape_functions, float)
        self.target = self.ref_location if target is None else np.asarray(target, float)
        self.weight = float(weight)

        variables = []
        for node in nodes:
            variables += [node.x, node.y, node.z]
        self.variables = variables

    @property
    def ref_location(self):
        return evaluate_ref(self.nodes, self.shape_functions[0])

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
