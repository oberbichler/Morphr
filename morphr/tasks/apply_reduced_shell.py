import morphr as mo

import anurbs as an
import eqlib as eq
import numpy as np

REDUCED_SHELL_3P = mo.ReducedIgaShell


class ApplyReducedShell(mo.Task):
    membrane_stiffness: float
    bending_stiffness: float
    weight: float = 1

    def run(self, config, job, data, log):
        cad_model = data.get('cad_model', None)
        model_tolerance = job.model_tolerance

        nb_objectives = 0

        # FIXME: Check for None

        data['nodes'] = data.get('nodes', {})
        elements = []

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

            for u, v, weight in an.integration_points(face, model_tolerance):
                nonzero_indices, shape_functions = surface_geometry.shape_functions_at(u, v, 2)

                element = REDUCED_SHELL_3P(nodes[nonzero_indices], self.membrane_stiffness, self.bending_stiffness)
                element.add(shape_functions, weight)

                elements.append(element)

                nb_objectives += 1

        data['elements'] = data.get('elements', [])
        data['elements'].append(('ReducedIgaShell', elements, self.weight))

        # output

        log.info(f'{len(elements)} elements with {nb_objectives} new objectives')
