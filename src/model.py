import numpy as np
from numpy import linalg as la
from scipy.linalg import expm

import matplotlib.pyplot as plt


class Nonneg_dagma():
    """
    Projected Gradient Descet algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def dagness(self, W):
        """
        Evaluates the acyclicity constraint
        """
        return self.N * np.log(self.s) - la.slogdet(self.s*self.Id - W)[1]
    
    def gradient_acyclic(self, W):
        return la.inv(self.s*self.Id - W).T

    def fit(self, X, alpha, lamb, stepsize, s=1, max_iters=1000, checkpoint=250, tol=1e-6,
            adam_opt=False, beta1=.99, beta2=.999, track_seq=False, verb=False):
        
        self.init_variables_(X, track_seq, s, adam_opt, beta1, beta2, verb)
        self.W_est, _ = self.proj_grad_desc_(lamb, alpha, stepsize, max_iters,
                                                    checkpoint, tol, track_seq)
        
        return self.W_est


    def init_variables_(self, X, track_seq, s, adam_opt, beta1, beta2, verb):
        self.M, self.N = X.shape
        self.Cx = X.T @ X / self.M
        self.W_est = np.zeros_like(self.Cx)
        self.verb = verb

        self.Id = np.eye(self.N)
        self.s = s

        # For Adam
        self.opt_m, self.opt_v = 0, 0
        self.adam = adam_opt
        self.beta1, self.beta2 = beta1, beta2
        
        self.acyclicity = []
        self.diff = []
        self.seq_W = [] if track_seq else None

    def compute_gradient_(self, W, lamb, alpha):
        G_loss = self.Cx @(W - self.Id) / 2 + lamb
        G_acyc = self.gradient_acyclic(W)
        return G_loss + alpha*G_acyc
    
    def compute_adam_grad_(self, grad, iter):
        self.opt_m = self.opt_m * self.beta1 + (1 - self.beta1) * grad
        self.opt_v = self.opt_v * self.beta2 + (1 - self.beta2) * (grad ** 2)
        m_hat = self.opt_m / (1 - self.beta1 ** iter)
        v_hat = self.opt_v / (1 - self.beta2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad


    def proj_grad_step_(self, W, alpha, lamb, stepsize, iter):
        G_obj_func = self.compute_gradient_(W, lamb, alpha)
        if self.adam:
            G_obj_func = self.compute_adam_grad_(G_obj_func, iter+1)
        W_est = np.maximum(W - stepsize*G_obj_func, 0)

        # Ensure non-negative acyclicity
        acyc = self.dagness(W_est)
        if acyc < -1e-12:
            eigenvalues, _ = np.linalg.eig(W_est)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            W_est = W_est/(max_eigenvalue + 1e-3)
            acyc = self.dagness(W_est)

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

    def proj_grad_desc_(self, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                        track_seq):
        W_prev = self.W_est.copy()
        for i in range(max_iters):
            self.W_est, stepsize = self.proj_grad_step_(W_prev, alpha, lamb, stepsize, i)

            # Update tracking variables
            norm_W_prev = la.norm(W_prev)
            norm_W_prev = norm_W_prev if norm_W_prev != 0 else 1
            self.diff.append(la.norm(self.W_est - W_prev) / norm_W_prev)
            if track_seq:
                self.seq_W.append(self.W_est)
                self.acyclicity.append(self.dagness(self.W_est))

            # Check convergence
            if i % checkpoint == 0 and self.diff[-1] <= tol:
                break
    
            W_prev = self.W_est.copy()
        
        return self.W_est, stepsize


class Nonneg_notears(Nonneg_dagma):
    """
    Projected Gradient Descet algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def fit(self, X, alpha, lamb, stepsize, max_iters=1000, checkpoint=250, tol=1e-6,
            track_seq=False, verb=False):
        
        self.init_variables_(X, track_seq, None, verb)
        self.W_est, _ = self.proj_grad_desc_(lamb, alpha, stepsize, max_iters,
                                                    checkpoint, tol, track_seq)
        
        return self.W_est

    def dagness(self, W):
        """
        Evaluates the acyclicity constraint
        """
        return np.trace(expm(W)) - self.N
    
    def gradient_acyclic(self, W):
        return expm(W).T

    def proj_grad_step_(self, W, alpha, lamb, stepsize):
        G_obj_func = self.compute_gradient_(W, lamb, alpha)
        W_est = np.maximum(W - stepsize*G_obj_func, 0)
        
        return W_est, stepsize

class MetMulDagma(Nonneg_dagma):
    """
    Method of ultipliers algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def fit(self, X, lamb, stepsize, s=1, iters_in=1000, iters_out=10, checkpoint=250, tol=1e-6,
            beta=5, gamma=.25, rho_0=1, alpha_0=.1, track_seq=False, dec_step=None,
            adam_opt=False, beta1=.99, beta2=.999, verb=False):

        self.init_variables_(X, rho_0, alpha_0, track_seq, s, adam_opt, beta1, beta2,  verb)        
        dagness_prev = self.dagness(self.W_est)

        for i in range(iters_out):
            # Estimate W
            self.W_est, stepsize = self.proj_grad_desc_(lamb, self.alpha, stepsize, iters_in,
                                                        checkpoint, tol, track_seq)

            # Update augmented Lagrangian parameters
            dagness = self.dagness(self.W_est)
            self.rho = beta*self.rho if dagness > gamma*dagness_prev else self.rho
            
            # Update Lagrange multiplier
            self.alpha += self.rho*dagness

            dagness_prev = dagness

            if dec_step:
                stepsize *= dec_step

            if verb:
                print(f'- {i+1}/{iters_out}. Diff: {self.diff[-1]:.6f} | Acycl: {dagness:.6f}' +
                      f' | Rho: {self.rho:.3f} - Alpha: {self.alpha:.3f} - Step: {stepsize:.4f}')
                                    
        return self.W_est  

    def compute_gradient_(self, W, lamb, alpha):
        acyc_val = self.dagness(W)
        G_loss = self.Cx @(W - self.Id) / 2 + lamb
        G_acyc = self.gradient_acyclic(W)
        return G_loss + (alpha + self.rho*acyc_val)*G_acyc
    
    def init_variables_(self, X, rho_init, alpha_init, track_seq, s, adam_opt, beta1, beta2,  verb):
        super().init_variables_(X, track_seq, s, adam_opt, beta1, beta2,  verb)
        self.rho = rho_init
        self.alpha = alpha_init

    
class BarrierDagma(Nonneg_dagma):
    def fit(self, X, lamb, stepsize, s=1, iters_in=1000, iters_out=10, checkpoint=250, tol=1e-6,
            beta=.5, delta=1e-5, alpha=1, track_seq=False, dec_step=False, verb=False):

        self.init_variables_(X, track_seq, verb)        
        self.delta = delta 

        for i in range(iters_out):
            # Estimate W
            self.W_est, stepsize = self.proj_grad_desc_(lamb, alpha, s, stepsize, iters_in,
                                                        checkpoint, tol, track_seq)

            # Logarithmic barrier weight
            alpha *= beta

            if dec_step:
                stepsize *= .9

            if verb:
                dagness = self.dagness(self.W_est, s)
                print(f'- {i+1}/{iters_out}. Diff: {self.diff[-1]:.4f} | Acycl: {dagness:.4f}' +
                      f' | Alpha: {alpha:.3f} - Step: {stepsize:.4f}')
                                    
        return self.W_est
    
    def compute_gradient_(self, W, lamb, s, alpha):
        dagness = np.maximum(self.dagness(W, s), self.delta) 
        G_loss = self.Cx @(W - self.Id) + lamb
        G_acyc = la.inv(s*self.Id - W).T
        return G_loss - alpha/dagness*G_acyc
    
