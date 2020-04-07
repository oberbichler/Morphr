from morphr import PointCoupling, RotationCoupling, Task
import numpy as np
import eqlib as eq
import anurbs as an


class ApplyEdgeCoupling(Task):
    penalty_displacement: float = 1.0
    penalty_rotation: float = 1.0

    def run(self, config, job, data, log):
        model_tolerance = job.model_tolerance
        cad_model = data.get('cad_model', None)
        penalty_displacement = self.penalty_displacement
        penalty_rotation = self.penalty_rotation

        # FIXME: Check for None

        nb_conditions = 0

        data['nodes'] = nodes = data.get('nodes', {})
        data['elements'] = elements = data.get('elements', [])

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

                nodes[nurbs_surface_a] = nurbs_surface_nodes
            else:
                nurbs_surface_nodes_a = nodes[nurbs_surface_a]

            if nurbs_surface_b not in nodes:
                nurbs_surface_nodes = []

                for x, y, z in nurbs_surface_b.poles:
                    nurbs_surface_nodes.append(eq.Node(x, y, z))

                nurbs_surface_nodes_b = np.array(nurbs_surface_nodes, object)

                nodes[nurbs_surface_b] = nurbs_surface_nodes
            else:
                nurbs_surface_nodes_b = nodes[nurbs_surface_b]

            integration_points_a, integration_points_b = an.integration_points(edge, tolerance=model_tolerance)

            for (t_a, weight), (t_b, _) in zip(integration_points_a, integration_points_b):
                u_a, v_a = trim_a.curve_geometry.data.point_at(t_a)
                u_b, v_b = trim_b.curve_geometry.data.point_at(t_b)

                indices_a, shape_functions_a = nurbs_surface_a.shape_functions_at(u_a, v_a, 1)
                indices_b, shape_functions_b = nurbs_surface_b.shape_functions_at(u_b, v_b, 1)

                element_nodes_a = [nurbs_surface_nodes_a[i] for i in indices_a]
                element_nodes_b = [nurbs_surface_nodes_b[i] for i in indices_b]

                if penalty_displacement != 0:
                    element = PointCoupling(element_nodes_a, element_nodes_b, shape_functions_a, shape_functions_b, weight * penalty_displacement)
                    elements.append(element)

                    nb_conditions += 1

                if penalty_rotation != 0:
                    _, t2_edge = trim_a.curve_3d.derivatives_at(t_a, order=1)
                    t2_edge /= np.linalg.norm(t2_edge)

                    element = RotationCoupling(element_nodes_a, element_nodes_b, shape_functions_a, shape_functions_b, t2_edge, weight * penalty_rotation)
                    elements.append(element)

                    nb_conditions += 1

                    # assert_almost_equal(act_lhs, exp_lhs)

                    if self.debug:
                        cad_model.add(an.Point3D(element.act_b), r'{"layer": "Debug/ApplyEdgeCoupling/RotationAxis"}')

                if self.debug:
                    point_a = nurbs_surface_a.point_at(u_a, v_a)
                    point_b = nurbs_surface_b.point_at(u_b, v_b)
                    cad_model.add(an.Point3D(point_a), r'{"layer": "Debug/ApplyEdgeCoupling/PointsA"}')
                    cad_model.add(an.Point3D(point_b), r'{"layer": "Debug/ApplyEdgeCoupling/PointsB"}')

                    if penalty_rotation != 0:
                        cad_model.add(an.Line3D(point_a, point_a + t2_edge), r'{"layer": "Debug/ApplyEdgeCoupling/RotationAxis"}')

        # output

        log.info(f'{nb_conditions} new conditions')
