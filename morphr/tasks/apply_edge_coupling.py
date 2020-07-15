import morphr as mo

import anurbs as an
import eqlib as eq
import numpy as np

from typing import Dict, Union

POINT_DISTANCE = eq.IgaPointDistanceAD
NORMAL_DISTANCE = eq.IgaRotationCouplingAD


class ApplyEdgeCoupling(mo.Task):
    projection_tolerance: float = 1e-8
    weight: Union[float, Dict[str, float]] = 1.0

    def run(self, config, job, data, log):
        model_tolerance = job.model_tolerance
        cad_model = data.get('cad_model', None)

        if isinstance(self.weight, float):
            weight_displacement = self.weight
            weight_rotation = self.weight
        else:
            weight_displacement = self.weight.get('displacement', 0)
            weight_rotation = self.weight.get('rotation', 0)

        # FIXME: Check for None

        nb_objectives = 0

        data['nodes'] = nodes = data.get('nodes', {})
        point_distance_group = []
        normal_distance_group = []

        for key, edge in cad_model.of_type('BrepEdge'):
            if edge.nb_trims != 2:
                continue

            (_, trim_a), (_, trim_b) = edge.trims

            nurbs_surface_key_a, nurbs_surface_a = trim_a.surface_geometry
            nurbs_surface_key_b, nurbs_surface_b = trim_b.surface_geometry

            if nurbs_surface_a not in nodes:
                nurbs_surface_nodes = []

                for x, y, z in nurbs_surface_a.poles:
                    nurbs_surface_nodes.append(eq.Node(x, y, z))

                nurbs_surface_nodes_a = np.array(nurbs_surface_nodes, object)

                nodes[nurbs_surface_a] = nurbs_surface_nodes_a
            else:
                nurbs_surface_nodes_a = nodes[nurbs_surface_a]

            if nurbs_surface_b not in nodes:
                nurbs_surface_nodes = []

                for x, y, z in nurbs_surface_b.poles:
                    nurbs_surface_nodes.append(eq.Node(x, y, z))

                nurbs_surface_nodes_b = np.array(nurbs_surface_nodes, object)

                nodes[nurbs_surface_b] = nurbs_surface_nodes_b
            else:
                nurbs_surface_nodes_b = nodes[nurbs_surface_b]

            integration_points_a, integration_points_b = an.integration_points(edge, tolerance=self.projection_tolerance, tessellation_tolerance=model_tolerance)

            for (t_a, weight), (t_b, _) in zip(integration_points_a, integration_points_b):
                u_a, v_a = trim_a.curve_geometry.data.point_at(t_a)
                u_b, v_b = trim_b.curve_geometry.data.point_at(t_b)

                indices_a, shape_functions_a = nurbs_surface_a.shape_functions_at(u_a, v_a, 1)
                indices_b, shape_functions_b = nurbs_surface_b.shape_functions_at(u_b, v_b, 1)

                element_nodes_a = [nurbs_surface_nodes_a[i] for i in indices_a]
                element_nodes_b = [nurbs_surface_nodes_b[i] for i in indices_b]

                if weight_displacement != 0:
                    element = POINT_DISTANCE(element_nodes_a, element_nodes_b)
                    element.add(shape_functions_a, shape_functions_b, weight * weight_displacement)
                    point_distance_group.append(element)

                    nb_objectives += 1

                if weight_rotation != 0:
                    _, axis = trim_a.curve_3d.derivatives_at(t_a, order=1)

                    element = NORMAL_DISTANCE(element_nodes_a, element_nodes_b)
                    element.add(shape_functions_a, shape_functions_b, axis, weight=weight * weight_rotation)
                    normal_distance_group.append(element)

                    nb_objectives += 1

                if self.debug:
                    point_a = nurbs_surface_a.point_at(u_a, v_a)
                    point_b = nurbs_surface_b.point_at(u_b, v_b)
                    cad_model.add(an.Point3D(point_a), r'{"layer": "Debug/ApplyEdgeCoupling/PointsA"}')
                    cad_model.add(an.Point3D(point_b), r'{"layer": "Debug/ApplyEdgeCoupling/PointsB"}')

                    if weight_rotation != 0:
                        cad_model.add(an.Line3D(point_a, point_a + axis), r'{"layer": "Debug/ApplyEdgeCoupling/RotationAxis"}')

        data['elements'] = data.get('elements', [])

        if weight_displacement != 0:
            data['elements'].append(('DisplacementCoupling', point_distance_group, weight_displacement))

        if weight_rotation != 0:
            data['elements'].append(('IgaRotationCouplingAD', normal_distance_group, weight_rotation))

        # output

        log.info(f'{len(point_distance_group) + len(normal_distance_group)} with {nb_objectives} new objectives')
