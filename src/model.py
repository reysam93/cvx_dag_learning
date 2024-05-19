import numpy as np
from numpy import linalg as la

import matplotlib.pyplot as plt


class Nonneg_dagma():
    """
    Projected Gradient Descet algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def dagness(self, W, s=1):
        """
        Evaluates the acyclicity constraint
        """
        return self.N * np.log(s) - la.slogdet(s*self.Id - W)[1]

    def fit(self, X, alpha, lamb, stepsize, s=1, max_iters=1000, checkpoint=250, tol=1e-4,
            track_seq=False, verb=False):
        
        self.init_variables_(X, track_seq, verb)
        self.W_est, _ = self.proj_grad_desc_(lamb, alpha, s, stepsize, max_iters,
                                                    checkpoint, tol, track_seq)
        
        return self.W_est


    def init_variables_(self, X, track_seq, verb):
        self.M, self.N = X.shape
        self.Cx = X.T @ X / self.M
        self.W_est = np.zeros_like(self.Cx)
        self.Id = np.eye(self.N)
        self.verb = verb

        self.acyclicity = []
        self.diff = []
        self.seq_W = [] if track_seq else None

    def compute_gradient_(self, W, lamb, s, alpha):
        G_loss = self.Cx @(W - self.Id) / 2 + lamb
        G_acyc = la.inv(s*self.Id - W).T
        return G_loss + alpha*G_acyc

    def proj_grad_step_(self, W, alpha, s, lamb, stepsize):
        # G_obj_func = self.G_loss_(W, lamb) + alpha*self.G_const_(W, s)
        G_obj_func = self.compute_gradient_(W, lamb, s, alpha)
        W_est = np.maximum(W - stepsize*G_obj_func, 0)

        # Ensure non-negative acyclicity
        acyc = self.dagness(W_est, s)
        if acyc < -1e-12:
            eigenvalues, _ = np.linalg.eig(W_est)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            W_est = W_est/(max_eigenvalue + 1e-3)
            acyc = self.dagness(W_est, s)

            stepsize /= 2
            if self.verb:
                print('Negative acyclicity. Projecting and reducing stepsize to: ', stepsize)

        assert acyc > -1e-12, f'Acyclicity is negative: {acyc}'
        
        return W_est, stepsize

    def convergence_(self, iteration, checkpoint, tol, W_prev):
        """
        Check if the algorithm has converged
        """
        if iteration % checkpoint == 0:
            self.diff.append(la.norm(self.W_est - W_prev) / la.norm(W_prev))
            return self.diff <= tol

        return False

    def proj_grad_desc_(self, lamb, alpha, s, stepsize, max_iters, checkpoint, tol,
                        track_seq):
        W_prev = self.W_est.copy()
        for i in range(max_iters):
            self.W_est, stepsize = self.proj_grad_step_(W_prev, alpha, s, lamb, stepsize)

            # Update tracking variables
            norm_W_prev = la.norm(W_prev)
            norm_W_prev = norm_W_prev if norm_W_prev != 0 else 1
            self.diff.append(la.norm(self.W_est - W_prev) / norm_W_prev)
            if track_seq:
                self.seq_W.append(self.W_est)
                self.acyclicity.append(self.dagness(self.W_est, s))

            # Check convergence
            if i % checkpoint == 0 and self.diff[-1] <= tol:
                break
    
            W_prev = self.W_est.copy()
        
        return self.W_est, stepsize
    

class MetMulDagma(Nonneg_dagma):
    """
    Method of ultipliers algorithm for learning DAGs with DAGMA acyclicity constraint
    """

    def fit(self, X, lamb, stepsize, s=1, iters_in=1000, iters_out=10, checkpoint=250, tol=1e-4,
            beta=3, gamma=.25, rho_0=1, alpha_0=.1, track_seq=False, verb=False):

        self.init_variables_(X, rho_0, alpha_0, track_seq, verb)        
        # dagness_prev = self.dagness(W_prev, s)
        dagness_prev = self.dagness(self.W_est, s)

        for i in range(iters_out):
            # Estimate W
            self.W_est, stepsize = self.proj_grad_desc_(lamb, self.alpha, s, stepsize, iters_in,
                                                        checkpoint, tol, track_seq)

            # Update augmented Lagrangian parameters
            dagness = self.dagness(self.W_est, s)
            self.rho = beta*self.rho if dagness > gamma*dagness_prev else self.rho
            
            # Update Lagrange multiplier
            self.alpha += self.rho*dagness

            dagness_prev = dagness
            stepsize *= .9

            # # Check convergence - REPEATED CODE
            # # if i % checkpoint == 0:
            # diff = la.norm(self.W_est - W_prev) / la.norm(W_prev)
            #     # if diff <= tol:
            #     #     break

            W_prev = self.W_est.copy()
            if verb:
                print(f'- {i+1}/{iters_out}. Diff: {self.diff[-1]:.4f} | Acycl: {dagness:.4f}' +
                      f' | Rho: {self.rho:.3f} - Alpha: {self.alpha:.3f} - Step: {stepsize:.4f}')
                                    
        return self.W_est  

    def compute_gradient_(self, W, lamb, s, alpha):
        acyc_val = self.dagness(W, s)
        G_loss = self.Cx @(W - self.Id) / 2 + lamb
        G_acyc = la.inv(s*self.Id - W).T
        return G_loss + (alpha + self.rho*acyc_val)*G_acyc
    
    def init_variables_(self, X, rho_init, alpha_init, track_seq, verb):
        super().init_variables_(X, track_seq, verb)
        self.rho = rho_init
        self.alpha = alpha_init

    # def minimize_(self, W, lamb, alpha, rho, s, stepsize):
    #     acyc_val = self.dagness(W, s)
    #     G_obj_func = self.G_loss_(W, lamb) + (alpha + rho*acyc_val) * self.G_const_(W, s)
    #     W_est = np.maximum(W - stepsize*G_obj_func, 0)

    #     # Ensure non-negative acyclicity
    #     acyc = self.dagness(W_est, s)
    #     if acyc < -1e-12:
    #         eigenvalues, _ = np.linalg.eig(W_est)
    #         max_eigenvalue = np.max(np.abs(eigenvalues))
    #         W_est = W_est/(max_eigenvalue + 1e-3)
    #         acyc = self.dagness(W_est, s)

    #         stepsize /= 2
    #         # if verb:
    #         #     print('Negative acyclicity. Projecting and reducing stepsize to: ', stepsize)

    #     assert acyc > -1e-12, f'Acyclicity is negative: {acyc}'
        
    #     return W_est, stepsize


    

