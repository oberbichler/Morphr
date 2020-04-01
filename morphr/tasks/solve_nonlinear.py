from morphr import PointSupport, Task
import numpy as np
import eqlib as eq
import anurbs as an


class SolveNonlinear(Task):
    max_iterations: int = 100
    damping: float = 0

    def run(self, config, job, data):
        elements = data.get('elements', None)

        problem = eq.Problem(elements, nb_threads=1)

        print(f'{len(elements)} conditions')
        print(f'{problem.nb_variables} variables')

        eq.Log.info_level = 5

        solver = eq.NewtonRaphson(problem)
        solver.maxiter = self.max_iterations
        solver.damping = self.damping

        solver.run()

        for surface, nodes in data['nodes'].items():
            for i, node in enumerate(nodes):
                surface.set_pole(i, node.act_location)
