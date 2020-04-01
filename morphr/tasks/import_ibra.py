from morphr import PointSupport, Task
import numpy as np
import eqlib as eq
import anurbs as an


class ImportIbra(Task):
    path: str

    def run(self, config, job, data):
        model = an.Model()
        model.load(self.path)

        data['cad_model'] = model
