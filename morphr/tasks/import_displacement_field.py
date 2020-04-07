from morphr import Task
import numpy as np
import meshio


class ImportDisplacementField(Task):
    mesh_0: str
    mesh_1: str

    def run(self, config, job, data, log):
        mesh_0 = meshio.read(self.mesh_0, file_format='obj')
        mesh_1 = meshio.read(self.mesh_1, file_format='obj')

        nb_points = len(mesh_0.points)

        vertices = np.empty((nb_points, 3), float)
        displacements = np.empty((nb_points, 3), float)

        for i, (point_0, point_1) in enumerate(zip(mesh_0.points, mesh_1.points)):
            vertices[i] = point_0
            displacements[i] = np.subtract(point_1, point_0)

        faces = []

        if 'triangle' in mesh_0.cells_dict:
            for a, b, c in mesh_0.cells_dict['triangle']:
                faces.append([a, b, c])

        if 'quad' in mesh_0.cells_dict:
            for a, b, c, d in mesh_0.cells_dict['quad']:
                faces.append([a, b, c])
                faces.append([c, d, a])

        data['vertices'] = vertices
        data['displacements'] = displacements
        data['faces'] = faces

        # output

        log.info(f'{len(vertices)} vertices')
        log.info(f'{len(faces)} triangles')
