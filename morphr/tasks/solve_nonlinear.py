from morphr import Task
import eqlib as eq
import numpy as np
import time


class SolveNonlinear(Task):
    r_tolerance: float = 1e-6
    max_iterations: int = 100
    damping: float = 0
    auto_scale: bool = False
    nb_threads: int = 1

    def run(self, config, job, data, log):
        element_groups = data.get('elements', None)

        elements = []

        for _, group_elements, _ in element_groups:
            elements.extend(group_elements)

        start_time = time.perf_counter()

        problem = eq.Problem(elements, nb_threads=self.nb_threads, grainsize=100)

        end_time = time.perf_counter()
        time_ellapsed = end_time - start_time
        log.benchmark(f'Assembly done in {time_ellapsed:.2f} sec')

        log.info(f'{len(elements)} objectives')
        log.info(f'{problem.nb_variables} variables')

        if self.auto_scale:
            self.solve_auto_scale(log, problem, elements, element_groups)
        else:
            self.solve(log, problem)

        for surface, nodes in data['nodes'].items():
            for i, node in enumerate(nodes):
                surface.poles[i] = node.act_location

    def solve(self, log, problem):
        for i in range(self.max_iterations):
            log.info(f'Iteration {i+1}/{self.max_iterations}...')

            problem.compute()

            if np.linalg.norm(problem.df) < self.r_tolerance:
                break

            if self.damping != 0:
                problem.hm_add_diagonal(self.damping)

            dx = problem.hm_inv_v(problem.df)

            problem.x -= dx

            log.info(f'rnorm = {np.linalg.norm(problem.df)}')
            log.info(f'xnorm = {np.linalg.norm(dx)}')

    def solve_auto_scale(self, log, problem, elements, element_groups):
        f = np.empty_like(problem.f, float)
        g = np.empty_like(problem.df, float)
        h = np.empty_like(problem.hm_values, float)

        scaling_factors = np.empty(len(element_groups), float)

        log.info(f'Problem consists of {len(element_groups)} groups')
        for i, group in enumerate(element_groups):
            log.info(f'  {i}: {group[0]} with {len(group[1])} elements')

        for iteration in range(self.max_iterations):
            log.info(f'Iteration {iteration+1}/{self.max_iterations}...')

            f.fill(0)
            g.fill(0)
            h.fill(0)

            for j, (group_name, group_elements, group_weight) in enumerate(element_groups):
                log.benchmark(f'Compute "{group_name}" ({len(group_elements)} @ {group_weight})...')

                for element in elements:
                    element.is_active = False

                for element in group_elements:
                    element.is_active = True

                start_time = time.perf_counter()

                problem.compute()

                end_time = time.perf_counter()
                time_ellapsed = end_time - start_time
                time_ellapsed_per_element = time_ellapsed / len(group_elements)
                log.benchmark(f'Computation of {group_name} in {time_ellapsed:.2f} sec')
                log.benchmark(f'{time_ellapsed_per_element:.5f} sec/element')

                if iteration == 0:
                    condition_norm_inf = problem.hm_norm_inf

                    scaling_factor = group_weight / condition_norm_inf

                    scaling_factors[j] = scaling_factor

                    problem.scale(scaling_factor)

                    # log.info(f'Norm of "{group_name}" = {condition_norm_inf}')
                    # log.info(f'after scaling = {problem.hm_norm_inf}')
                else:
                    problem.scale(scaling_factors[j])

                f += problem.f
                g += problem.df
                h += problem.hm_values

            problem.f = f
            problem.df[:] = g
            problem.hm_values[:] = h

            if np.linalg.norm(problem.df) < self.r_tolerance:
                log.info(f'rnorm = {np.linalg.norm(problem.df)}')
                break

            if self.damping != 0:
                problem.hm_add_diagonal(self.damping)

            dx = problem.hm_inv_v(g)

            problem.x -= dx

            log.info(f'rnorm = {np.linalg.norm(problem.df)}')
            log.info(f'xnorm = {np.linalg.norm(dx)}')

        for element in elements:
            element.is_active = True
