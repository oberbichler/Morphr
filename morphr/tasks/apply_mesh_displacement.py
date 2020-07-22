import morphr as mo

import anurbs as an
import eqlib as eq
import numpy as np

ELEMENT = eq.IgaPointLocation


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

        mapper = an.MeshMapper3D(vertices, faces)

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

                    success, min_location, (min_face, min_parameter), _ = mapper.map(location, max_distance)

                    if not success:
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
