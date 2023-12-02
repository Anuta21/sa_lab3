from sa_lab3.solve import Solve, SolveCustom
from sa_lab3.output import PolynomialBuilder, PolynomialBuilderCustom
import itertools
from concurrent import futures
from time import time
from stqdm import stqdm
import numpy as np

__author__ = 'nikiandr'

def coth(x):
    return  (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    # return  1 / (np.exp(-x) + 1)
    # return np.sin(x)


# CUSTOM_TRANSFORM = (np.tanh, 'tanh', 'arctanh')
CUSTOM_TRANSFORM = (coth, 'coth', 'arcth')

def print_stats(method_name, func_runtimes):
    if method_name not in func_runtimes:
        print("{!r} wasn't profiled, nothing to display.".format(method_name))
    else:
        runtimes = func_runtimes[method_name]
        total_runtime = sum(runtimes)
        average = total_runtime / len(runtimes)
        print('function: {!r}'.format(method_name))
        print(f'\trun times: {len(runtimes)}')
        print(f'\taverage run time: {average:.7f}')

def getError(params):
    params_new = params[-1].copy()
    params_new['degrees'] = [*(params[:-1])]
    if params_new['custom_structure']:
        solver = SolveCustom(params_new, CUSTOM_TRANSFORM)
    else:
        solver = Solve(params_new)
    func_runtimes = solver.prepare()
    normed_error = min(solver.norm_error)
    return (params_new['degrees'], normed_error, func_runtimes)

def getSolution(params, pbar_container, max_deg=15):
    if params['degrees'][0] == 0:
        x1_range = list(range(1, max_deg+1))
    else:
        x1_range = [params['degrees'][0]]
    
    if params['degrees'][1] == 0:
        x2_range = list(range(1, max_deg+1))
    else:
        x2_range = [params['degrees'][1]]
    
    if params['degrees'][2] == 0:
        x3_range = list(range(1, max_deg+1))
    else:
        x3_range = [params['degrees'][2]]

    ranges = list(itertools.product(x1_range, x2_range, x3_range, [params]))
    tick = time()
    if len(ranges) > 1:
        with futures.ThreadPoolExecutor() as pool:
            results = list(stqdm(
                pool.map(getError, ranges), 
                total=len(ranges), 
                st_container=pbar_container,
                desc='**Підбір степенів**',
                backend=True, frontend=False))

        results.sort(key=lambda t: t[1])
    else:
        results = [getError(ranges[0])]
    # func_runtimes = {key: [] for key in results[-1][-1].keys()}
    # for key in func_runtimes:
    #     for res in results:
    #         func_runtimes[key] += res[-1][key]

    final_params = params.copy()
    final_params['degrees'] = results[0][0]
    if final_params['custom_structure']:
        solver = SolveCustom(final_params, CUSTOM_TRANSFORM)
    else:
        solver = Solve(final_params)
    solver.prepare()
    tock = time()
    
    print('\n--- BEGIN DEBUG INFO ---')
    # for func in func_runtimes:
    #     print_stats(func, func_runtimes)

    print(f'TOTAL RUNTIME: {tock-tick:.3f} sec\n\n')

    solver.save_to_file()
    if final_params['custom_structure']:
        solution = PolynomialBuilderCustom(solver)
    else:
        solution = PolynomialBuilder(solver)
    
    return solver, solution, final_params['degrees']