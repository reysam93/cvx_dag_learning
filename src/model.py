import numpy as np
from numpy import linalg as la
from scipy.linalg import expm

class Nonneg_dagma():
    """
    Projected Gradient Descet algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def __init__(self, primal_opt='pgd', acyclicity='logdet'):
        self.acyc_const = acyclicity
        if acyclicity == 'logdet':
            self.dagness = self.logdet_acyc_
            self.gradient_acyclic = self.logdet_acyclic_grad_
        elif acyclicity == 'matexp':
            self.dagness = self.matexp_acyc_
            self.gradient_acyclic = self.matexp_acyclic_grad_
        else:
            raise ValueError('Unknown acyclicity constraint')

        self.opt_type = primal_opt
        if primal_opt in ['pgd', 'adam']:
            self.minimize_primal = self.proj_grad_desc_
        elif primal_opt == 'fista':
            self.minimize_primal = self.acc_proj_grad_desc_
        else:
            raise ValueError('Unknown solver type for primal problem')

    def logdet_acyc_(self, W):
        """
        Evaluates the acyclicity constraint
        """
        return self.N * np.log(self.s) - la.slogdet(self.s*self.Id - W)[1]
    
    def logdet_acyclic_grad_(self, W):
        return la.inv(self.s*self.Id - W).T
    
    def matexp_acyc_(self, W):
        # Clip W to prevent overflowing
        entry_limit = np.maximum(10, 5e2/W.shape[0])
        W = np.clip(W, -entry_limit, entry_limit)
        return np.trace(expm(W)) - self.N

    def matexp_acyclic_grad_(self, W):
        # Clippling gradient to prevent overflow
        return np.clip(expm(W).T, -1e7, 1e7)



    def fit(self, X, alpha, lamb, stepsize, s=1, max_iters=1000, checkpoint=250, tol=1e-6,
            beta1=.99, beta2=.999, Sigma=1, track_seq=False, verb=False):
        
        self.init_variables_(X, track_seq, s, Sigma, beta1, beta2, verb)
        self.W_est, _ = self.minimize_primal(self.W_est, lamb, alpha, stepsize, max_iters,
                                             checkpoint, tol, track_seq)
        
        return self.W_est

    def init_variables_(self, X, track_seq, s, Sigma, beta1, beta2, verb):
        self.M, self.N = X.shape
        self.Cx = X.T @ X / self.M
        self.W_est = np.zeros_like(self.Cx)
        self.verb = verb

        if np.isscalar(Sigma):
            self.Sigma_inv = 1 / Sigma * np.ones((self.N))
        elif Sigma.ndim == 1:
            self.Sigma_inv = 1 / Sigma
        elif Sigma.ndim == 2:
            assert np.all(Sigma == np.diag(np.diag(Sigma))), 'Sigma must be a diagonal matrix'
            self.Sigma_inv = 1 / np.diag(Sigma)
        else:
            raise ValueError("Sigma must be a scalar, vector or diagonal Matrix")

        self.Id = np.eye(self.N)
        self.s = s

        # For Adam
        self.opt_m, self.opt_v = 0, 0
        self.beta1, self.beta2 = beta1, beta2
        
        self.acyclicity = []
        self.diff = []
        self.seq_W = [] if track_seq else None

    def compute_gradient_(self, W, lamb, alpha):        
        G_loss = self.Cx @(W - self.Id) * self.Sigma_inv / 2 + lamb
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
        if self.opt_type == 'adam':
            G_obj_func = self.compute_adam_grad_(G_obj_func, iter+1)
        W_est = np.maximum(W - stepsize*G_obj_func, 0)

        # Ensure non-negative acyclicity
        if self.acyc_const == 'logdet':
            acyc = self.dagness(W_est)        
            if acyc < -1e-12:
                eigenvalues, _ = np.linalg.eig(W_est)
                max_eigenvalue = np.max(np.abs(eigenvalues))
                W_est = W_est/(max_eigenvalue + 1e-2)
                acyc = self.dagness(W_est)

                stepsize /= 2
                if self.verb:
                    print('Negative acyclicity. Projecting and reducing stepsize to: ', stepsize)

                assert acyc > -1e-12, f'Acyclicity is negative: {acyc}'
        
        return W_est, stepsize

    def tack_variables_(self, W, W_prev, track_seq):
        norm_W_prev = la.norm(W_prev)
        norm_W_prev = norm_W_prev if norm_W_prev != 0 else 1
        self.diff.append(la.norm(W - W_prev) / norm_W_prev)
        if track_seq:
            self.seq_W.append(W)
            self.acyclicity.append(self.dagness(W))


    def proj_grad_desc_(self, W, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                        track_seq):
        W_prev = W.copy()
        for i in range(max_iters):
            W, stepsize = self.proj_grad_step_(W_prev, alpha, lamb, stepsize, i)

            # Update tracking variables
            self.tack_variables_(W, W_prev, track_seq)

            # Check convergence
            if i % checkpoint == 0 and self.diff[-1] <= tol:
                break
    
            W_prev = W.copy()
        
        return W, stepsize

    def acc_proj_grad_desc_(self, W, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                            track_seq):
        W_prev = W.copy()
        W_fista = np.copy(W) 
        t_k = 1
        for i in range(max_iters):
            W, stepsize = self.proj_grad_step_(W_fista, alpha, lamb, stepsize, i)
            t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
            W_fista = W + (t_k - 1)/t_next*(W - W_prev)

            # Update tracking variables
            self.tack_variables_(W, W_prev, track_seq)

            # Check convergence
            if i % checkpoint == 0 and self.diff[-1] <= tol:
                break

            W_prev = W
            t_k = t_next
        
        return W, stepsize



class MetMulDagma(Nonneg_dagma):
    """
    Method of ultipliers algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def fit(self, X, lamb, stepsize, s=1, iters_in=1000, iters_out=10, checkpoint=250, tol=1e-6,
            beta=5, gamma=.25, rho_0=1, alpha_0=.1, track_seq=False, dec_step=None,
            beta1=.99, beta2=.999, Sigma=1, verb=False):

        self.init_variables_(X, rho_0, alpha_0, track_seq, s, Sigma, beta1, beta2,  verb)        
        dagness_prev = self.dagness(self.W_est)

        for i in range(iters_out):
            # Estimate W
            self.W_est, stepsize = self.minimize_primal(self.W_est, lamb, self.alpha, stepsize, iters_in,
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
        G_loss = self.Cx @(W - self.Id) * self.Sigma_inv / 2 + lamb
        acyc_val = self.dagness(W)
        G_acyc = self.gradient_acyclic(W)
        return G_loss + (alpha + self.rho*acyc_val)*G_acyc

    def init_variables_(self, X, rho_init, alpha_init, track_seq, s, Sigma, beta1, beta2,  verb):
        super().init_variables_(X, track_seq, s, Sigma, beta1, beta2,  verb)
        self.rho = rho_init
        self.alpha = alpha_init
