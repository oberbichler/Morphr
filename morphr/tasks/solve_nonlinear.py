from morphr import Task
import eqlib as eq
import numpy as np
import scipy.sparse.linalg as la
import time


def inf_norm(sparse_matrix):
    return la.norm(sparse_matrix, np.inf)


class SolveNonlinear(Task):
    max_iterations: int = 100
    damping: float = 0
    auto_scale: bool = False
    nb_threads: int = 1

    def run(self, config, job, data, log):
        elements = data.get('elements', None)

        problem = eq.Problem(elements, nb_threads=self.nb_threads)

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
        element_types = set([type(element) for element in elements])

        f = np.empty_like(problem.f, float)
        g = np.empty_like(problem.df, float)
        h = np.empty_like(problem.hm_values, float)

        scaling_factors = np.empty(len(element_types), float)

        for i in range(self.max_iterations):
            log.info(f'Iteration {i+1}/{self.max_iterations}...')

            f.fill(0)
            g.fill(0)
            h.fill(0)

            for j, element_type in enumerate(element_types):
                nb_elements = 0

                for element in elements:
                    element.is_active = isinstance(element, element_type)
                    if element.is_active:
                        nb_elements += 1

                start_time = time.perf_counter()

                problem.compute()

                end_time = time.perf_counter()
                time_ellapsed = end_time - start_time
                time_ellapsed_per_element = time_ellapsed / nb_elements
                log.info(f'Computation of {element_type.__name__} in {time_ellapsed:.2f} sec')
                log.info(f'{time_ellapsed_per_element:.5f} sec/element')

                if i == 0:
                    lhs = problem.general_hm

                    condition_norm_inf = inf_norm(lhs)

                    scaling_factor = 1 / condition_norm_inf

                    scaling_factors[j] = scaling_factor

                    log.info(f'Norm {element_type.__name__} = {condition_norm_inf}')
                    log.info(f'Norm {element_type.__name__} = {inf_norm(lhs * scaling_factor)}')
                else:
                    scaling_factor = scaling_factors[j]

                if element_type.__name__ == 'NormalDistance':
                    scaling_factor *= 1e-4

                f += problem.f * scaling_factor
                g += problem.df * scaling_factor
                h += problem.hm_values * scaling_factor

            problem.f = f
            problem.df[:] = g
            problem.hm_values[:] = h

            if self.damping != 0:
                problem.hm_add_diagonal(self.damping)

            dx = problem.hm_inv_v(g)

            problem.x -= dx

            log.info(f'rnorm = {np.linalg.norm(problem.df)}')
            log.info(f'xnorm = {np.linalg.norm(dx)}')

        for element in elements:
            element.is_active = True
