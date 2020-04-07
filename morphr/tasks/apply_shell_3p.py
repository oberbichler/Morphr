from morphr import Shell3P, Task
import numpy as np
import eqlib as eq
import anurbs as an


class ApplyShell3P(Task):
    thickness: float
    youngs_modulus: float
    poissons_ratio: float

    def run(self, config, job, data, log):
        cad_model = data.get('cad_model', None)
        model_tolerance = job.model_tolerance

        nb_conditions = 0

        # FIXME: Check for None

        data['nodes'] = data.get('nodes', {})

        data['elements'] = elements = data.get('elements', [])

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

            for u, v, weight in an.integration_points(face, model_tolerance):
                nonzero_indices, shape_functions = surface_geometry.shape_functions_at(u, v, 2)

                element = Shell3P(nodes[nonzero_indices], shape_functions, thickness, youngs_modulus, poissons_ratio, weight)
                elements.append(element)

                nb_conditions += 1

        # output

        log.info(f'{nb_conditions} new conditions')
