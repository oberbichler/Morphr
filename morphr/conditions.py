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


