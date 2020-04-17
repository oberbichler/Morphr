from morphr import Task
import anurbs as an


class ImportIbra(Task):
    path: str

    def run(self, config, job, data, log):
        model = an.Model()
        model.load(self.path)

        data['cad_model'] = model

        # output

        nb_faces = len(model.of_type('BrepFace'))
        nb_edges = len(model.of_type('BrepEdge'))

        log.info(f'{nb_faces} faces')
        log.info(f'{nb_edges} edges')
