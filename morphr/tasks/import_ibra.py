from morphr import Task
import anurbs as an


class ImportIbra(Task):
    path: str

    def run(self, config, job, data):
        model = an.Model()
        model.load(self.path)

        data['cad_model'] = model

        # output

        nb_faces = len(model.of_type('BrepFace'))

        print(f'{nb_faces} faces')
