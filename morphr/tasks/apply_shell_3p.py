import morphr as mo

import anurbs as an
import eqlib as eq
import numpy as np

ELEMENT = eq.IgaShell3PAD


class ApplyShell3P(mo.Task):
    thickness: float
    youngs_modulus: float
    poissons_ratio: float
    weight: float = 1

    def run(self, config, job, data, log):
        cad_model = data.get('cad_model', None)
        model_tolerance = job.model_tolerance

        nb_objectives = 0

        # FIXME: Check for None

        data['nodes'] = data.get('nodes', {})
        elements = []

        thickness = self.thickness
        youngs_modulus = self.youngs_modulus
        poissons_ratio = self.poissons_ratio

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

            for span_u, span_v, integration_points in an.integration_points_with_spans(face, model_tolerance):
                nonzero_indices = surface_geometry.nonzero_pole_indices_at_span(span_u, span_v)

                element = ELEMENT(nodes[nonzero_indices], thickness, youngs_modulus, poissons_ratio)
                elements.append(element)

                for u, v, weight in integration_points:
                    _, shape_functions = surface_geometry.shape_functions_at(u, v, 2)

                    element.add(shape_functions, weight)

                    nb_objectives += 1

        data['elements'] = data.get('elements', [])
        data['elements'].append(('IgaShell3PAD', elements, self.weight))

        # output

        log.info(f'{len(elements)} elements with {nb_objectives} new objectives')
