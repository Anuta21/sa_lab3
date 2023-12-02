import numpy as np
from numpy.polynomial import Polynomial as pnm
from solve import Solve
import basis_gen as b_gen


class PolynomialBuilder(object):
    def __init__(self, solution):
        assert isinstance(solution, Solve)
        self._solution = solution
        max_degree = max(solution.deg) - 1
        if solution.poly_type == 'Поліноми Лежандра':
            self.symbol = 'T'
            # self.basis = b_gen.basis_legendre(max_degree)
        elif solution.poly_type == 'Зсунуті поліноми Лежандра':
            self.symbol = 'U^*'
            self.basis = b_gen.basis_sh_legendre(max_degree)
        elif solution.poly_type == 'Синус':
            self.symbol = 'sin'
        assert self.symbol
        self.a = solution.a.T.tolist()
        self.c = solution.c.T.tolist()
        self.minX = [X.min(axis=0).ravel() for X in solution.X_]
        self.maxX = [X.max(axis=0).ravel() for X in solution.X_]
        self.minY = solution.Y_.min(axis=0).ravel()
        self.maxY = solution.Y_.max(axis=0).ravel()

    def _form_lamb_lists(self):
        """
        Generates specific basis coefficients for Psi functions
        """
        self.lamb = list()
        for i in range(self._solution.Y.shape[1]):  # `i` is an index for Y
            lamb_i = list()
            shift = 0
            for j in range(3):  # `j` is an index to choose vector from X
                lamb_i_j = list()
                for k in range(self._solution.dim[j]):  # `k` is an index for vector component
                    lamb_i_jk = self._solution.Lamb[shift:shift + self._solution.deg[j], i].ravel()
                    shift += self._solution.deg[j]
                    lamb_i_j.append(lamb_i_jk)
                lamb_i.append(lamb_i_j)
            self.lamb.append(lamb_i)

    def _print_psi_i_jk(self, i, j, k):
        """
        Returns string of Psi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :param k: an index for vector component
        :return: result string
        """
        strings = list()
        for n in range(len(self.lamb[i][j][k])):
            if np.abs(self.lamb[i][j][k][n]) > 1e-7:
                if self.symbol == 'cos' or self.symbol == 'sin':
                    strings.append(r'(1 + \{symbol}(2 \pi \cdot {deg}\cdot x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                        self.lamb[i][j][k][n], j+1, k+1,
                        symbol=self.symbol, deg=n))
                else:
                    strings.append(r'(1 + {symbol}_{{{deg}}}(x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                        self.lamb[i][j][k][n], j+1, k+1,
                        symbol=self.symbol, deg=n))
        res = r' \cdot '.join(strings)
        return res.replace('+ -', '- ')

    def _print_phi_i_j(self, i, j):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """
        strings = list()
        for k in range(len(self.lamb[i][j])):
            shift = sum(self._solution.dim[:j]) + k
            for n in range(len(self.lamb[i][j][k])):
                if np.abs(self.a[i][shift] * self.lamb[i][j][k][n]) > 1e-7:
                    if self.symbol == 'cos' or self.symbol == 'sin':
                        strings.append(r'(1 + \{symbol}(2 \pi \cdot {deg}\cdot x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                            self.a[i][shift] * self.lamb[i][j][k][n],
                            j+1, k+1, symbol=self.symbol, deg=n))
                    else:
                        strings.append(r'(1 + {symbol}_{{{deg}}}(x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                            self.a[i][shift] * self.lamb[i][j][k][n],
                            j+1, k+1, symbol=self.symbol, deg=n))
        res = r' \cdot '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                for n in range(len(self.lamb[i][j][k])):
                    if np.abs(self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n]) > 1e-7:
                        if self.symbol == 'cos' or self.symbol == 'sin':
                            strings.append(r'(1 + \{symbol}(2 \pi \cdot {deg}\cdot x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                                self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n],
                                j + 1, k + 1, symbol=self.symbol, deg=n))
                        else:
                            strings.append(r'(1 + {symbol}_{{{deg}}}(x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                                self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n],
                                j + 1, k + 1, symbol=self.symbol, deg=n))
        res = r' \cdot '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i_F_ij(self, i):
        res = []
        for j in range(3):
            coef = self.c[i][j]
            res.append(f'(1 + \\Phi_{{{i+1}{j+1}}} (x_{j+1}))^{{{coef:.6f}}}')
        return '\cdot'.join(res) + ' - 1'

    def get_results(self):
        """
        Generates results based on given solution
        :return: Results string
        """
        self._form_lamb_lists()
        psi_strings = [r'$\Psi^{{{0}}}_{{[{1},{2}]}} (x_{{{1}{2}}}) = {result} - 1$'.format(i + 1, j + 1, k + 1,
                                                                 result=self._print_psi_i_jk(i, j, k)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)
                       for k in range(self._solution.dim[j])]
        phi_strings = [r'$\Phi_{{{0}{1}}} (x_{{{1}}}) = {result} - 1$'.format(i + 1, j + 1, result=self._print_phi_i_j(i, j)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_strings = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result} - 1$'.format(i + 1, result=self._print_F_i(i)) + '\n'
                     for i in range(self._solution.Y.shape[1])]
        f_strings_from_f_ij = [r'$\Phi_{i}(x_1, x_2, x_3) = {result}$'.format(i=i+1, result=self._print_F_i_F_ij(i)) + '\n' 
                                for i in range(self._solution.Y.shape[1])]
        return '\n'.join(
            [r'Функції $\Phi_i$ через $\Phi_{i1}(x_1)$, $\Phi_{i2}(x_2)$, $\Phi_{i3}(x_3)$:' + '\n'] + f_strings_from_f_ij +
            [r'Функції $\Phi_i$:' + '\n'] + f_strings + 
            [r'Функції $\Phi_{ik}$:' + '\n'] + phi_strings + 
            [r'Функції $\Psi$:' + '\n'] + psi_strings)

class PolynomialBuilderCustom(PolynomialBuilder):
    def __init__(self, solution):
        """
        solution - data for Solve init,
        """
        super().__init__(solution)
        self.nonlinear_func = solution.nonlinear_func_name
        self.nonlinear_func_inv = solution.nonlinear_func_inv_name

    def _print_psi_i_jk(self, i, j, k, mode=0):
        strings = list()
        for n in range(len(self.lamb[i][j][k])):
            if np.abs(self.lamb[i][j][k][n]) > 1e-7:
                if self.symbol == 'cos' or self.symbol == 'sin':
                    strings.append(r'(1 + \mathrm{{{func}}}(2 \pi \cdot \{symbol}({deg} \cdot x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                        self.lamb[i][j][k][n], j+1, k+1,
                        symbol=self.symbol, deg=n, func=self.nonlinear_func))
                else:
                    strings.append(r'(1 + \mathrm{{{func}}}({symbol}_{{{deg}}}(x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                        self.lamb[i][j][k][n], j+1, k+1,
                        symbol=self.symbol, deg=n, func=self.nonlinear_func))
        res = r' \cdot '.join(strings)
        return res.replace('+ -', '- ')

    def _print_phi_i_j(self, i, j, mode=0):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """
        strings = list()
        for k in range(len(self.lamb[i][j])):
            shift = sum(self._solution.dim[:j]) + k
            for n in range(len(self.lamb[i][j][k])):
                if np.abs(self.a[i][shift] * self.lamb[i][j][k][n]) > 1e-7:
                    if self.symbol == 'cos' or self.symbol == 'sin':
                        strings.append(r'(1 + \mathrm{{{func}}}(2 \pi \cdot \{symbol}({deg} \cdot x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                            self.a[i][shift] * self.lamb[i][j][k][n],
                            j+1, k+1, symbol=self.symbol, deg=n, func=self.nonlinear_func))
                    else:                    
                        strings.append(r'(1 + \mathrm{{{func}}}({symbol}_{{{deg}}}(x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                            self.a[i][shift] * self.lamb[i][j][k][n],
                            j+1, k+1, symbol=self.symbol, deg=n, func=self.nonlinear_func))
        res = r' \cdot '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i(self, i, mode=0):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                for n in range(len(self.lamb[i][j][k])):
                    if np.abs(self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n]) > 1e-7:
                        if self.symbol == 'cos' or self.symbol == 'sin':
                            strings.append(r'(1 + \mathrm{{{func}}}(2 \pi \cdot \{symbol}({deg} \cdot x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                                self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n],
                                j + 1, k + 1, symbol=self.symbol, deg=n, func=self.nonlinear_func))
                        else:
                            strings.append(r'(1 + \mathrm{{{func}}}({symbol}_{{{deg}}}(x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                                self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n],
                                j + 1, k + 1, symbol=self.symbol, deg=n, func=self.nonlinear_func))
        res = r' \cdot '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i_F_ij(self, i):
        res = []
        for j in range(3):
            coef = self.c[i][j]
            res.append(f'(1 + \\mathrm{{{self.nonlinear_func}}}(\\Phi_{{{i+1}{j+1}}} (x_{j+1})))^{{{coef:.6f}}}')
        return '\cdot'.join(res) + ' - 1'

    def get_results(self):
        """
        Generates results based on given solution
        :return: Results string
        """
        self._form_lamb_lists()
        psi_strings = [r'$\Psi^{{{0}}}_{{[{1},{2}]}} = \mathrm{{{inv_func}}}[{result} - 1]$'.format(i + 1, j + 1, k + 1,
                       inv_func=self.nonlinear_func_inv,    
                       result=self._print_psi_i_jk(i, j, k)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)
                       for k in range(self._solution.dim[j])]
        phi_strings = [r'$\Phi_{{{0}{1}}} = \mathrm{{{inv_func}}}[{result} - 1]$'.format(i + 1, j + 1, 
                       inv_func=self.nonlinear_func_inv,
                       result=self._print_phi_i_j(i, j)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_strings = [r'$\Phi_{{{0}}} = \mathrm{{{inv_func}}}[{result} - 1]$'.format(i + 1, 
                     inv_func=self.nonlinear_func_inv,
                     result=self._print_F_i(i)) + '\n'
                     for i in range(self._solution.Y.shape[1])]
        f_strings_from_f_ij = [r'$\Phi_{i}(x_1, x_2, x_3) = \mathrm{{{inv_func}}}[{result}]$'.format(i=i+1, 
                                inv_func=self.nonlinear_func_inv,
                                result=self._print_F_i_F_ij(i)) + '\n' 
                                for i in range(self._solution.Y.shape[1])]
        return '\n'.join(
            [r'Функції $\Phi_i$ через $\Phi_{i1}(x_1)$, $\Phi_{i2}(x_2)$, $\Phi_{i3}(x_3)$:' + '\n'] + f_strings_from_f_ij +
            [r'Функції $\Phi_i$:' + '\n'] + f_strings + 
            [r'Функції $\Phi_{ik}$:' + '\n'] + phi_strings + 
            [r'Функції $\Psi$:' + '\n'] + psi_strings)
