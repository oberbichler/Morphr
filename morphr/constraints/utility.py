import numpy as np
import hyperjet as hj


def normSq(v):
    return np.dot(v, v)


def norm(v):
    return np.dot(v, v)**0.5


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
