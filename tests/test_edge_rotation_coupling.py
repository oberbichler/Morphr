import json
import os
import pytest
import eqlib as eq
from numpy.testing import assert_almost_equal
from morphr.conditions import EdgeRotationCoupling

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)


@pytest.fixture()
def test_data():
    directory = os.path.dirname(__file__)
    with open(os.path.join(directory, r'test_edge_rotation_coupling_data.json'), 'r') as f:
        return json.load(f)


def test_edge_rotation_coupling(test_data):
    for data in test_data:
        nodes_a = []
        nodes_b = []

        for ref_location, act_location in zip(data['ref_locations_a'], data['act_locations_a']):
            node = eq.Node(*ref_location)
            node.act_location = act_location
            nodes_a.append(node)

        for ref_location, act_location in zip(data['ref_locations_b'], data['act_locations_b']):
            node = eq.Node(*ref_location)
            node.act_location = act_location
            nodes_b.append(node)

        element = EdgeRotationCoupling(
            nodes_a,
            nodes_b,
            data['shape_functions_a'],
            data['shape_functions_b'],
            data['t2_edge'],
            data['weight'],
        )

        f, g, h = element.compute_all()

        assert_almost_equal(f, data['exp_f'])
        assert_almost_equal(g, data['exp_g'])
        assert_almost_equal(h, data['exp_h'])
