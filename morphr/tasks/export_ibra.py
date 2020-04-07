from morphr import Task


class ExportIbra(Task):
    path: str

    def run(self, config, job, data, log):
        model = data.get('cad_model', None)

        if model is None:
            raise RuntimeError('No CAD model available')

        model.save(self.path)
