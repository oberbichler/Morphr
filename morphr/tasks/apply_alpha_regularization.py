from morphr import PointNodeCoupling, Task
import numpy as np
import eqlib as eq
import anurbs as an


class ApplyAlphaRegularization(Task):
    penalty: float = 1.0

    def run(self, config, job, data, log):
        cad_model = data.get('cad_model', None)

        # FIXME: Check for None

        data['nodes'] = data.get('nodes', {})

        data['elements'] = elements = data.get('elements', [])

        nb_conditions = 0

        for key, face in cad_model.of_type('BrepFace'):
            surface_geometry_key = surface_geometry = face.surface_geometry.data

            if surface_geometry_key not in data['nodes']:
                nodes = []

                for x, y, z in surface_geometry.poles:
                    nodes.append(eq.Node(x, y, z))
                nodes = np.array(nodes, object)
                data['nodes'][surface_geometry_key] = nodes
            else:
                nodes = data['nodes'][surface_geometry_key]

            for r in range(surface_geometry.nb_poles_u):
                for s in range(surface_geometry.nb_poles_v):
                    target_node = nodes[r * surface_geometry.nb_poles_v + s]
                    u, v = surface_geometry.greville_point(r, s)

                    nonzero_indices, shape_functions = surface_geometry.shape_functions_at(u, v, 0)

                    element = PointNodeCoupling(nodes[nonzero_indices], shape_functions, target_node, self.penalty)
                    elements.append(element)

                    nb_conditions += 1

                    if self.debug:
                        cad_model.add(an.Line3D(target_node.act_location, surface_geometry.point_at(u, v)), r'{"layer": "Debug/ApplyAlphaRegularization/Connections"}')

        # output

        log.info(f'{nb_conditions} new conditions')
