import morphr as mo

import anurbs as an
import eqlib as eq
import numpy as np

ELEMENT = eq.IgaPointLocation


def squared_distance(v):
    return np.dot(v, v)


def line_projection(point, a, b):
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


class MeshMapper:
    def __init__(self, vertices, faces, max_distance):
        self.vertices = np.asarray(vertices, float)
        self.faces = np.asarray(faces, int)
        self.max_distance_squared = float(max_distance)**2

        rtree = an.RTree3D(len(faces))

        for face in faces:
            face_vertices = vertices[face]

            box_min = np.min(face_vertices, axis=0)
            box_max = np.max(face_vertices, axis=0)

            rtree.add(box_min, box_max)

        rtree.finish()

        self.rtree = rtree

    def closest_point(self, sample):
        min_distance_squared = float('inf')
        min_location = None
        min_mesh_parameter = (None, None)

        for face_index in self.rtree.by_point(sample, self.max_distance_squared**0.5):
            face = self.faces[face_index]

            a, b, c = np.take(self.vertices, face, axis=0)

            closest_point, closest_parameter = an.Triangle3D.projection(sample, a, b, c)

            if np.min(closest_parameter) < 0 or np.max(closest_parameter) > 1:
                closest_point_ab, closest_parameter_ab = line_projection(sample, a, b)
                closest_point_bc, closest_parameter_bc = line_projection(sample, b, c)
                closest_point_ca, closest_parameter_ca = line_projection(sample, c, a)

                distance_squared = squared_distance(closest_point_ab - sample)
                closest_point = closest_point_ab
                closest_parameter = [closest_parameter_ab, 1 - closest_parameter_ab, 0]

                if squared_distance(closest_point_bc - sample) < distance_squared:
                    distance_squared = squared_distance(closest_point_bc - sample)
                    closest_point = closest_point_bc
                    closest_parameter = [0, closest_parameter_bc, 1 - closest_parameter_bc]

                if squared_distance(closest_point_ca - sample) < distance_squared:
                    distance_squared = squared_distance(closest_point_ca - sample)
                    closest_point = closest_point_ca
                    closest_parameter = [0, 1 - closest_parameter_ca, closest_parameter_ca]
            else:
                d = np.subtract(closest_point, sample)
                distance_squared = d.dot(d)

            if distance_squared > self.max_distance_squared or distance_squared > min_distance_squared:
                continue

            min_distance_squared = distance_squared
            min_location = closest_point
            min_mesh_parameter = (face_index, closest_parameter)

        return min_location, min_mesh_parameter


class ApplyMeshDisplacement(mo.Task):
    max_distance: float = 0
    weight: float = 1

    def run(self, config, job, data, log):
        cad_model = data.get('cad_model', None)
        vertices = data.get('vertices', None)
        displacements = data.get('displacements', None)
        faces = data.get('faces', None)
        model_tolerance = job.model_tolerance
        max_distance = model_tolerance * 2 if self.max_distance <= 0 else self.max_distance

        nb_objectives = 0

        # FIXME: Check for None

        data['nodes'] = data.get('nodes', {})
        elements = []

        mapper = MeshMapper(vertices, faces, max_distance)

        projection_failed = 0        

        for i, (key, face) in enumerate(cad_model.of_type('BrepFace')):
            surface_geometry_key = surface_geometry = face.surface_geometry.data

            if surface_geometry_key not in data['nodes']:
                nodes = []

                for x, y, z in surface_geometry.poles:
                    nodes.append(eq.Node(x, y, z))

                nodes = np.array(nodes, object)

                data['nodes'][surface_geometry_key] = nodes
            else:
                nodes = data['nodes'][surface_geometry_key]

            for span_u, span_v, integration_points in an.integration_points_with_spans(face, model_tolerance):
                nonzero_indices = surface_geometry.nonzero_pole_indices_at_span(span_u, span_v)

                element = ELEMENT(nodes[nonzero_indices])
                elements.append(element)

                for u, v, weight in integration_points:
                    location = surface_geometry.point_at(u, v)

                    if self.debug:
                        cad_model.add(an.Point3D(location), r'{"layer": "Debug/ApplyMeshDisplacement/IntegrationPoints"}')

                    min_location, (min_face, min_parameter) = mapper.closest_point(location)

                    if min_face is None:
                        cad_model.add(an.Point3D(location), r'{"layer": "Debug/ApplyMeshDisplacement/Failed"}')
                        projection_failed += 1
                        continue

                    abc = faces[min_face]

                    dabc = displacements[abc]

                    displacement = np.dot(min_parameter, dabc)

                    n, shape_functions = surface_geometry.shape_functions_at(u, v, 0)

                    location_source = min_location
                    location_target = min_location + displacement

                    if self.debug:
                        cad_model.add(an.Point3D(location_source), r'{"layer": "Debug/ApplyMeshDisplacement/Source"}')
                        cad_model.add(an.Point3D(location_target), r'{"layer": "Debug/ApplyMeshDisplacement/Target"}')
                        cad_model.add(an.Line3D(location_source, location_target), r'{"layer": "Debug/ApplyMeshDisplacement/DisplacementField"}')
                        cad_model.add(an.Line3D(location, location_source), r'{"layer": "Debug/ApplyMeshDisplacement/Projection"}')

                    element.add(shape_functions, location_target, weight * self.weight)

                    nb_objectives += 1

        data['elements'] = data.get('elements', [])
        data['elements'].append(('MeshDisplacement', elements, self.weight))

        # output

        if projection_failed > 0:
            log.warning(f'Projection failed for {projection_failed} points')

        log.info(f'{len(elements)} elements with {nb_objectives} new objectives')
