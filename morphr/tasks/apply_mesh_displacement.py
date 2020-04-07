from morphr import PointSupport, Task
import numpy as np
import eqlib as eq
import anurbs as an


class ApplyMeshDisplacement(Task):
    penalty: float = 1

    def line_projection(self, point, a, b):
        dif = b - a
        dot = np.dot(dif, dif)

        if dot < 1e-14:
            return a, 0.0

        o = a
        r = dif / dot
        o2pt = point - o
        t = np.dot(o2pt, r)

        if t < 0:
            return a, 0.0

        if t > 1:
            return b, 1.0

        closest_point = o + dif * t

        return closest_point, t

    def run(self, config, job, data, log):
        cad_model = data.get('cad_model', None)
        vertices = data.get('vertices', None)
        displacements = data.get('displacements', None)
        faces = data.get('faces', None)
        penalty = self.penalty
        model_tolerance = job.model_tolerance

        nb_conditions = 0

        # FIXME: Check for None

        data['nodes'] = data.get('nodes', {})

        data['elements'] = elements = data.get('elements', [])

        rtree = an.RTree3D(len(faces))

        for face in faces:
            vabc = vertices[face]

            box_min = np.min(vabc, axis=0)
            box_max = np.max(vabc, axis=0)

            if self.debug:
                cad_model.add(an.Box3D(box_min, box_max), r'{"layer": "Debug/ApplyMeshDisplacement/Boxes"}')

            rtree.add(box_min, box_max)

        rtree.finish()

        for key, face in cad_model.of_type('BrepFace'):
            surface_geometry_key = surface_geometry = face.surface_geometry.data

            span_nodes = dict()
            span_data = dict()

            if surface_geometry_key not in data['nodes']:
                nodes = []

                for x, y, z in surface_geometry.poles:
                    nodes.append(eq.Node(x, y, z))

                nodes = np.array(nodes, object)

                data['nodes'][surface_geometry_key] = nodes
            else:
                nodes = data['nodes'][surface_geometry_key]

            for u, v, weight in an.integration_points(face, model_tolerance):
                location = surface_geometry.point_at(u, v)

                if self.debug:
                    cad_model.add(an.Point3D(location), r'{"layer": "Debug/ApplyMeshDisplacement/IntegrationPoints"}')

                indices = rtree.by_point(location, model_tolerance)

                min_distance2 = float('inf')
                min_location = None
                min_parameter = None
                min_face = None

                for index in indices:
                    a, b, c = faces[index]

                    va, vb, vc = vertices[[a, b, c]]

                    closest_point, parameter = an.Triangle3D.projection(location, va, vb, vc)

                    if np.min(parameter) < 0 or np.max(parameter) > 1:
                        continue

                    d = closest_point - location
                    distance2 = d.dot(d)

                    if distance2 > min_distance2:
                        continue

                    min_distance2 = distance2
                    min_location = closest_point
                    min_parameter = parameter
                    min_face = index

                    if distance2 < model_tolerance**2:
                        break

                if min_face is None:
                    continue

                abc = faces[min_face]

                dabc = displacements[abc]

                displacement = min_parameter.dot(dabc)

                nonzero_indices, shape_functions = surface_geometry.shape_functions_at(u, v, 0)

                location_source = min_location
                location_target = min_location + displacement

                element_data = [shape_functions, min_location - displacement, weight * penalty]

                span_u = an.upper_span(surface_geometry.degree_u, surface_geometry.knots_u, u)
                span_v = an.upper_span(surface_geometry.degree_v, surface_geometry.knots_v, v)

                span = (span_u, span_v)

                old_data = span_data.get(span, [])

                if len(old_data) == 0:
                    span_nodes[span] = nodes[nonzero_indices]

                span_data[span] = old_data + [element_data]

                if self.debug:
                    cad_model.add(an.Point3D(location_source), r'{"layer": "Debug/ApplyMeshDisplacement/ClosestPoints"}')
                    cad_model.add(an.Line3D(location_source, location_target), r'{"layer": "Debug/ApplyMeshDisplacement/DisplacementFields"}')
                    cad_model.add(an.Line3D(location, location_source), r'{"layer": "Debug/ApplyMeshDisplacement/Projections"}')

                element = PointSupport(nodes[nonzero_indices], shape_functions, min_location + displacement, weight * penalty)
                elements.append(element)

                nb_conditions += 1

        # output

        log.info(f'{nb_conditions} new conditions')
