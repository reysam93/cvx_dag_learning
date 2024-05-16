import numpy as np
from numpy import linalg as la

import matplotlib.pyplot as plt


class Nonneg_dagma():
    def dagness(self, W, s=1):
        """
        Evaluates the acyclicity constraint
        """
        return self.N * np.log(s) - la.slogdet(s*self.Id - W)[1]

    def minimize_(self, W, alpha, s, lamb, stepsize, verb):
        G_loss = self.Cx @(W - self.Id)
        G_constr = la.inv(s*self.Id - W).T
        G_obj_func = G_loss/2 + alpha*G_constr + lamb*np.ones_like(W)
        W_est = np.maximum(W - stepsize*G_obj_func, 0)

        # Ensure non-negative acyclicity
        acyc = self.dagness(W_est, s)
        if acyc < -1e-12:
            eigenvalues, _ = np.linalg.eig(W_est)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            W_est = W_est/(max_eigenvalue + 1e-3)
            acyc = self.dagness(W_est, s)

            stepsize /= 2
            if verb:
                print('Negative acyclicity. Projecting and reducing stepsize to: ', stepsize)

        assert acyc > -1e-12, f'Acyclicity is negative: {acyc}'
        
        return W_est, stepsize

    def fit(self, X, alpha, lamb, stepsize, s=1, max_iters=1000, checkpoint=250, tol=1e-4,
            W_true=None, verb=False):
        self.M, self.N = X.shape
        self.X = X
        
        self.Cx = X.T @ X / self.M
        self.W_est = np.zeros_like(self.Cx)
        W_prev = np.ones_like(self.Cx)
        self.Id = np.eye(self.N)

        norm_W_true = None if W_true is None else la.norm(W_true)
        self.errs_W = np.zeros(max_iters)
        self.acyclicity = np.zeros(max_iters)

        for iter in range(max_iters):
            self.W_est, stepsize = self.minimize_(self.W_est.copy(), alpha, s, lamb, stepsize,
                                                 verb)

            if W_true is not None:
                self.errs_W[iter] = (la.norm(W_true - self.W_est)/norm_W_true)**2
                self.acyclicity[iter] = self.dagness(self.W_est, s)

            # Check convergence
            if iter % checkpoint == 0:
                diff = la.norm(self.W_est - W_prev) / la.norm(W_prev)
                if diff <= tol:
                    break

                W_prev = self.W_est.copy()
                
        return self.W_est