from morphr import Task


class ExportMdpa(Task):
    path: str

    def run(self, config, job, data, log):
        vertices = data.get('vertices', None)
        displacements = data.get('displacements', None)
        faces = data.get('faces', None)

        with open(self.path, 'w') as f:
            f.write('Begin ModelPartData\n')
            f.write('End ModelPartData\n')
            f.write('\n')
            f.write('Begin Properties 0\n')
            f.write('End Properties\n')
            f.write('\n')
            f.write('Begin Nodes\n')
            for i, (x, y, z) in enumerate(vertices):
                f.write(f'  {i+1} {x} {y} {z}\n')
            f.write('End Nodes\n')
            f.write('\n')
            f.write('Begin Elements Element3D3N\n')
            for i, (a, b, c) in enumerate(faces):
                f.write(f'  {i+1} 0 {a+1} {b+1} {c+1}\n')
            f.write('End Elements\n')
            f.write('\n')
            f.write('Begin NodalData SHAPE_CHANGE_X\n')
            for i, u in enumerate(displacements[:, 0]):
                f.write(f'  {i+1} 0 {u}\n')
            f.write('End NodalData\n')
            f.write('\n')
            f.write('Begin NodalData SHAPE_CHANGE_Y\n')
            for i, u in enumerate(displacements[:, 1]):
                f.write(f'  {i+1} 0 {u}\n')
            f.write('End NodalData\n')
            f.write('\n')
            f.write('Begin NodalData SHAPE_CHANGE_Z\n')
            for i, u in enumerate(displacements[:, 2]):
                f.write(f'  {i+1} 0 {u}\n')
            f.write('End NodalData\n')
