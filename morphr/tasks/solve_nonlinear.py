from morphr import Task
import eqlib as eq
import numpy as np
import scipy.sparse.linalg as la


def inf_norm(sparse_matrix):
    return la.norm(sparse_matrix, np.inf)


class SolveNonlinear(Task):
    max_iterations: int = 100
    damping: float = 0
    auto_scale: bool = False

    def run(self, config, job, data, log):
        elements = data.get('elements', None)

        problem = eq.Problem(elements, nb_threads=1)

        log.info(f'{len(elements)} conditions')
        log.info(f'{problem.nb_variables} variables')

        if self.auto_scale:
            self.solve_auto_scale(log, problem, elements)
        else:
            self.solve(log, problem)

        for surface, nodes in data['nodes'].items():
            for i, node in enumerate(nodes):
                surface.set_pole(i, node.act_location)

    def solve(self, log, problem):
        for i in range(self.max_iterations):
            log.info(f'Iteration {i+1}/{self.max_iterations}...')

            problem.compute()

            if self.damping != 0:
                problem.hm_add_diagonal(self.damping)

            dx = problem.hm_inv_v(problem.df)

            problem.x -= dx

            log.info(f'rnorm = {np.linalg.norm(problem.df)}')
            log.info(f'xnorm = {np.linalg.norm(dx)}')

    def solve_auto_scale(self, log, problem, elements):
        from morphr import PointSupport

        system_element_types = [PointSupport]
        condition_element_types = set([type(element) for element in elements if type(element) not in system_element_types])

        f = np.zeros_like(problem.f)
        g = np.zeros_like(problem.df)
        h = np.zeros_like(problem.hm_values)

        for i in range(self.max_iterations):
            log.info(f'Iteration {i+1}/{self.max_iterations}...')

            for element in elements:
                element.is_active = type(element) in system_element_types

            problem.compute()

            system_norm_inf = inf_norm(problem.hm)

            f += problem.f
            g += problem.df
            h += problem.hm_values

            log.info(f'Norm System = {system_norm_inf}')

            for element_type in condition_element_types:
                for element in elements:
                    element.is_active = isinstance(element, element_type)

                problem.compute()

                condition_norm_inf = inf_norm(problem.hm)

                factor = system_norm_inf / condition_norm_inf

                f += problem.f * factor
                g += problem.df * factor
                h += problem.hm_values * factor

                log.info(f'Norm {element_type.__name__} = {condition_norm_inf}')

            problem.f = f
            problem.df[:] = g
            problem.hm_values[:] = h

            if self.damping != 0:
                problem.hm_add_diagonal(system_norm_inf * self.damping)

            dx = problem.hm_inv_v(g)

            problem.x -= dx

            log.info(f'rnorm = {np.linalg.norm(problem.df)}')
            log.info(f'xnorm = {np.linalg.norm(dx)}')

        for element in elements:
            element.is_active = True
