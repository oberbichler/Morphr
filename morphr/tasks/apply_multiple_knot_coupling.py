import morphr as mo

import anurbs as an
import eqlib as eq
import numpy as np

POINT_DISTANCE = eq.IgaPointDistanceAD


def _nonzero_spans(degree, knots):
    nonzero_spans = []

    for i in range(degree - 1, len(knots) - degree + 1):
        if knots[i] != knots[i + 1]:
            nonzero_spans.append((i, knots[i], knots[i + 1]))

    return nonzero_spans


def _get_multiple_knots(degree, knots):
    multiple_knots = []

    i = degree

    while i < len(knots) - degree:
        j = i + 1

        while j < len(knots) - degree and knots[i] == knots[j]:
            j += 1

        multiplicity = j - i

        if multiplicity != 1:
            multiple_knots.append((knots[i], i - 1, j - 1))

        i = j

    return multiple_knots


class ApplyMultipleKnotCoupling(mo.Task):
    weight: float = 1.0

    def run(self, config, job, data, log):
        model_tolerance = job.model_tolerance
        cad_model = data.get('cad_model', None)

        # FIXME: Check for None

        nb_objectives = 0

        data['nodes'] = nodes = data.get('nodes', {})
        group = []

        for key, face in cad_model.of_type('BrepFace'):
            _, surface_geometry = face.surface_geometry

            if surface_geometry not in nodes:
                surface_nodes = []

                for x, y, z in surface_geometry.poles:
                    surface_nodes.append(eq.Node(x, y, z))

                surface_nodes = np.array(surface_nodes, object)

                nodes[surface_geometry] = surface_nodes
            else:
                surface_nodes = nodes[surface_geometry]

            for u, span_u_a, span_u_b in _get_multiple_knots(surface_geometry.degree_u, surface_geometry.knots_u):
                for (span_v, v0, v1) in _nonzero_spans(surface_geometry.degree_v, surface_geometry.knots_v):
                    for v, weight in an.integration_points(surface_geometry.degree_v + 1, an.Interval(v0, v1)):
                        nonzero_indices_a, shape_functions_a = surface_geometry.shape_functions_at_span(u=u, v=v, span_u=span_u_a, span_v=span_v, order=1)
                        nonzero_indices_b, shape_functions_b = surface_geometry.shape_functions_at_span(u=u, v=v, span_u=span_u_b, span_v=span_v, order=1)

                        element_nodes_a = surface_nodes[nonzero_indices_a]
                        element_nodes_b = surface_nodes[nonzero_indices_b]

                        element = POINT_DISTANCE(element_nodes_a, element_nodes_b)

                        element.add([shape_functions_a[1]], [shape_functions_b[1]], weight * self.weight)

                        group.append(element)

                        nb_objectives += 1

                        if self.debug:
                            point = surface_geometry.point_at(u, v)
                            cad_model.add(an.Point3D(point), r'{"layer": "Debug/ApplyMultipleKnotCoupling/PointsU"}')

            for v, span_v_a, span_v_b in _get_multiple_knots(surface_geometry.degree_v, surface_geometry.knots_v):
                for (span_u, u0, u1) in _nonzero_spans(surface_geometry.degree_u, surface_geometry.knots_u):
                    for u, weight in an.integration_points(surface_geometry.degree_u + 1, an.Interval(u0, u1)):
                        nonzero_indices_a, shape_functions_a = surface_geometry.shape_functions_at_span(u=u, v=v, span_u=span_u, span_v=span_v_a, order=1)
                        nonzero_indices_b, shape_functions_b = surface_geometry.shape_functions_at_span(u=u, v=v, span_u=span_u, span_v=span_v_b, order=1)

                        element_nodes_a = surface_nodes[nonzero_indices_a]
                        element_nodes_b = surface_nodes[nonzero_indices_b]

                        element = POINT_DISTANCE(element_nodes_a, element_nodes_b)

                        element.add([shape_functions_a[2]], [shape_functions_b[2]], weight * self.weight)

                        group.append(element)

                        nb_objectives += 1

                        if self.debug:
                            point = surface_geometry.point_at(u, v)
                            cad_model.add(an.Point3D(point), r'{"layer": "Debug/ApplyMultipleKnotCoupling/PointsV"}')

        data['elements'] = data.get('elements', [])

        data['elements'].append(('DisplacementCoupling', group, self.weight))

        # output

        log.info(f'{len(group)} elements with {nb_objectives} new objectives')
