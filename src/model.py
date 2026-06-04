import numpy as np
from numpy import linalg as la
from scipy.linalg import expm


def _project_nonnegative_zero_diag(W):
    W_proj = np.maximum(W, 0)
    np.fill_diagonal(W_proj, 0)
    return W_proj


# NONNEGATIVE DAG LEARNING
class Nonneg_dagma():
    """
    Projected Gradient Descet algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def __init__(self, primal_opt='pgd', acyclicity='logdet', restart=True):
        self.acyc_const = acyclicity
        if acyclicity == 'logdet':
            self.dagness = self.logdet_acyc_
            self.gradient_acyclic = self.logdet_acyclic_grad_
        elif acyclicity == 'matexp':
            self.dagness = self.matexp_acyc_
            self.gradient_acyclic = self.matexp_acyclic_grad_
        else:
            raise ValueError('Unknown acyclicity constraint')

        self.restart = restart
        self.opt_type = primal_opt
        if primal_opt in ['pgd', 'adam']:
            self.minimize_primal = self.proj_grad_desc_
        elif primal_opt == 'fista':
            self.minimize_primal = self.acc_proj_grad_desc_
        elif primal_opt == 'sca' and isinstance(self, MetMulColide):
            # The subclass MetMulColide will handle this case 
            self.minimize_primal = None
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
            beta1=.99, beta2=.999, Sigma=1, delta=.01, track_seq=False,
            track_diagnostics=False, step_type='fixed', local_lipschitz_scale=1.0,
            min_stepsize=1e-12, max_stepsize=None, domain_bt_factor=0.5,
            domain_bt_max_iters=20, domain_bt_tol=1e-12, verb=False):
        
        self.init_variables_(X, track_seq, track_diagnostics, s, Sigma, beta1, beta2, delta, verb)
        self.configure_step_rule_(step_type, local_lipschitz_scale, min_stepsize,
                                  max_stepsize, domain_bt_factor,
                                  domain_bt_max_iters, domain_bt_tol)
        self.W_est, _ = self.minimize_primal(self.W_est, Sigma, lamb, alpha, stepsize, max_iters,
                                             checkpoint, tol, track_seq)
        self.record_diagnostics_(self.W_est, self.Sigma_est, lamb, alpha, None, stepsize, 0)
        
        return self.W_est
        
    def init_variables_(self, X, track_seq, track_diagnostics, s, Sigma, beta1, beta2, delta, verb):
        self.Gw_obj_func = None # for restart
        self.M, self.N = X.shape
        self.Cx = X.T @ X / self.M

        self.W_est = np.zeros_like(self.Cx)
        self.verb = verb

        if np.isscalar(Sigma) or Sigma.ndim == 1:
            self.Sigma_est = np.array(Sigma)
        elif Sigma.ndim == 2:
            assert np.all(Sigma == np.diag(np.diag(Sigma))), 'Sigma must be a diagonal matrix'
            self.Sigma_est = np.diag(Sigma)
        else:
            raise ValueError("Sigma must be a scalar, vector or diagonal Matrix")

        # For the logdet acyclicity
        self.Id = np.eye(self.N)
        self.s = s
        self.delta = delta

        # For Adam
        self.opt_m, self.opt_v = 0, 0
        self.beta1, self.beta2 = beta1, beta2
        
        # For tracking sequences
        self.acyclicity = []
        self.diff_W = []
        self.seq_W = [] if track_seq else None

        # For diagnostics
        self.track_diagnostics = track_diagnostics
        self.diagnostics = []
        self._fista_restarts_current = 0
        self._last_inner_iters = 0
        self._last_grad_norm = np.nan
        self._last_prox_grad_norm = np.nan
        self._last_eval_w_min = np.nan
        self._last_eval_w_neg_count = 0
        self._last_eval_sigma_min = np.nan
        self._last_stepsize_used = np.nan
        self._last_local_lipschitz = np.nan
        self._last_domain_bt_iters = 0
        self._cx_spectral_norm = la.norm(self.Cx, 2)

    def compute_score_(self, W, Sigma, lamb):
        D = self.Id - W
        quad_diag = np.diag(D.T @ self.Cx @ D)
        if np.isscalar(Sigma) or np.ndim(Sigma) == 0:
            score = 0.25 * np.sum(quad_diag) / Sigma
        else:
            score = 0.25 * np.sum(quad_diag / Sigma)
        score += lamb * np.sum(W)
        return score

    def augmented_lagrangian_value_(self, W, Sigma, lamb, alpha, rho=None):
        h_val = self.dagness(W)
        obj = self.compute_score_(W, Sigma, lamb) + alpha*h_val
        if rho is not None:
            obj += 0.5*rho*h_val**2
        return obj

    def record_diagnostics_(self, W, Sigma, lamb, alpha, rho, stepsize, outer_iter):
        if not self.track_diagnostics:
            return

        if self.acyc_const == 'logdet':
            domain_mat = self.s*self.Id - W
            slogdet_sign, slogdet_logabs = la.slogdet(domain_mat)
            spectral_radius = np.max(np.abs(la.eigvals(W))) if W.size else 0.0
        else:
            slogdet_sign, slogdet_logabs, spectral_radius = np.nan, np.nan, np.nan

        self.diagnostics.append({
            'outer_iter': outer_iter,
            'h': self.dagness(W),
            'rho': rho,
            'alpha': alpha,
            'stepsize': stepsize,
            'spectral_radius': spectral_radius,
            'slogdet_sign': slogdet_sign,
            'slogdet_logabs': slogdet_logabs,
            'diag_norm': la.norm(np.diag(W)),
            'grad_norm': self._last_grad_norm,
            'prox_grad_norm': self._last_prox_grad_norm,
            'diff_W': self.diff_W[-1] if self.diff_W else np.nan,
            'inner_iters': self._last_inner_iters,
            'fista_restarts': self._fista_restarts_current,
            'eval_w_min': self._last_eval_w_min,
            'eval_w_neg_count': self._last_eval_w_neg_count,
            'eval_sigma_min': self._last_eval_sigma_min,
            'stepsize_used': self._last_stepsize_used,
            'local_lipschitz': self._last_local_lipschitz,
            'domain_bt_iters': self._last_domain_bt_iters,
            'step_type': getattr(self, 'step_type', 'fixed'),
            'aug_lagrangian': self.augmented_lagrangian_value_(W, Sigma, lamb, alpha, rho),
        })

    def compute_gradient_W_(self, W, Sigma, lamb, alpha):
        G_loss = self.Cx @(W - self.Id) / Sigma / 2 + lamb
        G_acyc = self.gradient_acyclic(W)
        return G_loss + alpha*G_acyc
    
    def compute_adam_grad_(self, grad, opt_m, opt_v, iter):
        opt_m = opt_m * self.beta1 + (1 - self.beta1) * grad
        opt_v = opt_v * self.beta2 + (1 - self.beta2) * (grad ** 2)
        m_hat = opt_m / (1 - self.beta1 ** iter)
        v_hat = opt_v / (1 - self.beta2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad, opt_m, opt_v

    def configure_step_rule_(self, step_type='fixed', local_lipschitz_scale=1.0,
                             min_stepsize=1e-12, max_stepsize=None,
                             domain_bt_factor=0.5, domain_bt_max_iters=20,
                             domain_bt_tol=1e-12):
        valid_step_types = {
            'fixed',
            'local_lipschitz',
            'domain_backtracking',
            'local_lipschitz_domain_backtracking',
        }
        if step_type not in valid_step_types:
            raise ValueError(f'Unknown step_type: {step_type}')
        if domain_bt_factor <= 0 or domain_bt_factor >= 1:
            raise ValueError('domain_bt_factor must be in (0, 1)')
        self.step_type = step_type
        self.local_lipschitz_scale = local_lipschitz_scale
        self.min_stepsize = min_stepsize
        self.max_stepsize = max_stepsize
        self.domain_bt_factor = domain_bt_factor
        self.domain_bt_max_iters = domain_bt_max_iters
        self.domain_bt_tol = domain_bt_tol

    def uses_local_lipschitz_(self):
        return getattr(self, 'step_type', 'fixed') in {
            'local_lipschitz',
            'local_lipschitz_domain_backtracking',
        }

    def uses_domain_backtracking_(self):
        return getattr(self, 'step_type', 'fixed') in {
            'domain_backtracking',
            'local_lipschitz_domain_backtracking',
        }

    def clip_stepsize_(self, stepsize):
        stepsize = max(stepsize, self.min_stepsize)
        if self.max_stepsize is not None:
            stepsize = min(stepsize, self.max_stepsize)
        return stepsize

    def loss_lipschitz_(self, Sigma):
        if np.isscalar(Sigma) or np.ndim(Sigma) == 0:
            sigma_min = float(Sigma)
        else:
            sigma_min = np.min(Sigma)
        sigma_min = max(sigma_min, 1e-12)
        return self._cx_spectral_norm / (2*sigma_min)

    def logdet_terms_(self, W):
        A = self.s*self.Id - W
        sign, logabs = la.slogdet(A)
        h_val = self.N*np.log(self.s) - logabs
        G_acyc = la.inv(A).T
        grad_h_norm_sq = la.norm(G_acyc, 'fro')**2
        return h_val, G_acyc, grad_h_norm_sq, sign

    def logdet_domain_ok_(self, W):
        sign, logabs = la.slogdet(self.s*self.Id - W)
        h_val = self.N*np.log(self.s) - logabs
        return sign > 0 and h_val >= -self.domain_bt_tol

    def apply_domain_backtracking_(self, W, grad, stepsize):
        bt_iters = 0
        W_est = _project_nonnegative_zero_diag(W - stepsize*grad)
        while (bt_iters < self.domain_bt_max_iters
               and not self.logdet_domain_ok_(W_est)):
            stepsize = self.clip_stepsize_(stepsize*self.domain_bt_factor)
            W_est = _project_nonnegative_zero_diag(W - stepsize*grad)
            bt_iters += 1
        self._last_domain_bt_iters = bt_iters
        return W_est, stepsize

    def compute_gradient_and_stepsize_(self, W, Sigma, lamb, alpha, stepsize):
        self._last_local_lipschitz = np.nan
        if self.acyc_const == 'logdet' and self.uses_local_lipschitz_():
            G_loss = self.Cx @(W - self.Id) / Sigma / 2 + lamb
            _, G_acyc, grad_h_norm_sq, _ = self.logdet_terms_(W)
            grad = G_loss + alpha*G_acyc
            lipschitz = self.loss_lipschitz_(Sigma) + abs(alpha)*grad_h_norm_sq
            self._last_local_lipschitz = lipschitz
            stepsize = self.clip_stepsize_(self.local_lipschitz_scale / max(lipschitz, 1e-12))
            return grad, stepsize

        return self.compute_gradient_W_(W, Sigma, lamb, alpha), stepsize

    def proj_grad_step_W_(self, W, Sigma, alpha, lamb, stepsize, iter):
        self._last_eval_w_min = np.min(W)
        self._last_eval_w_neg_count = np.count_nonzero(W < 0)
        self._last_domain_bt_iters = 0
        self.Gw_obj_func, step_used = self.compute_gradient_and_stepsize_(
            W, Sigma, lamb, alpha, stepsize
        )
        return_stepsize = step_used
        if self.uses_local_lipschitz_() or self.uses_domain_backtracking_():
            return_stepsize = stepsize
        self._last_grad_norm = la.norm(self.Gw_obj_func)
        if self.opt_type == 'adam':
            self.Gw_obj_func, self.opt_m, self.opt_v = self.compute_adam_grad_(self.Gw_obj_func, self.opt_m, 
                                                                              self.opt_v, iter+1)
        if self.acyc_const == 'logdet' and self.uses_domain_backtracking_():
            W_est, step_used = self.apply_domain_backtracking_(W, self.Gw_obj_func, step_used)
        else:
            W_est = _project_nonnegative_zero_diag(W - step_used*self.Gw_obj_func)

        # Ensure non-negative acyclicity
        if self.acyc_const == 'logdet':
            acyc = self.dagness(W_est)        
            if acyc < -1e-12:
                eigenvalues, _ = np.linalg.eig(W_est)
                max_eigenvalue = np.max(np.abs(eigenvalues))
                # W_est = W_est/(max_eigenvalue + self.delta)
                W_est = (self.s - self.delta) * W_est / max_eigenvalue
                np.fill_diagonal(W_est, 0)
                acyc = self.dagness(W_est)

                step_used /= 2
                if self.verb:
                    print('Negative acyclicity. Projecting and reducing stepsize to: ', step_used)

                assert acyc > -1e-12, f'Acyclicity is negative: {acyc}'
        
        if not (self.uses_local_lipschitz_() or self.uses_domain_backtracking_()):
            return_stepsize = step_used
        self._last_prox_grad_norm = la.norm(W_est - W) / step_used if step_used > 0 else np.nan
        self._last_stepsize_used = step_used
        return W_est, return_stepsize

    def track_W_variable_(self, W, W_prev, track_seq):
        norm_W_prev = la.norm(W_prev)
        norm_W_prev = norm_W_prev if norm_W_prev != 0 else 1
        self.diff_W.append(la.norm(W - W_prev) / norm_W_prev)
        if track_seq:
            self.seq_W.append(W)
            self.acyclicity.append(self.dagness(W))

    def proj_grad_desc_(self, W, Sigma, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                        track_seq):
        W_prev = W.copy()
        for i in range(max_iters):
            W, stepsize = self.proj_grad_step_W_(W_prev, Sigma, alpha, lamb, stepsize, i)

            # Update tracking variables
            self.track_W_variable_(W, W_prev, track_seq)

            # Check convergence
            if i % checkpoint == 0 and self.diff_W[-1] <= tol:
                break
    
            W_prev = W.copy()
        
        self._last_inner_iters = i + 1
        return W, stepsize

    def acc_proj_grad_desc_(self, W, Sigma, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                            track_seq):
        W_prev = W.copy()
        W_fista = W.copy()
        t_k = 1  
        for i in range(max_iters):
            W, stepsize = self.proj_grad_step_W_(W_fista, Sigma, alpha, lamb, stepsize, i)
            diff_W = W - W_prev

            # Check if restarting condition is met
            if self.restart and np.vdot(self.Gw_obj_func, diff_W) > 1e-6:
                self._fista_restarts_current += 1
                W = W_prev.copy()                    
                W_fista = W.copy()
                t_k = 1
                continue

            t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
            W_fista = W + (t_k - 1)/t_next*(diff_W)

            # Update tracking variables
            self.track_W_variable_(W, W_prev, track_seq)

            # Check convergence
            if i % checkpoint == 0 and self.diff_W[-1] <= tol:
                break

            W_prev = W
            t_k = t_next

        self._last_inner_iters = i + 1
        return W, stepsize


# METMULDAGLEARNING
class MetMulDagma(Nonneg_dagma):
    """
    Method of ultipliers algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def fit(self, X, lamb, stepsize, s=1, iters_in=1000, iters_out=10, checkpoint=250, tol=1e-6,
            beta=5, gamma=.25, rho_0=1, alpha_0=.1, track_seq=False, dec_step=None,
            track_diagnostics=False, h_tol=1e-4, step_type='fixed',
            local_lipschitz_scale=1.0, min_stepsize=1e-12,
            max_stepsize=None, domain_bt_factor=0.5,
            domain_bt_max_iters=20, domain_bt_tol=1e-12,
            beta1=.99, beta2=.999, Sigma=1, delta=.01, verb=False):

        self.init_variables_(X, rho_0, alpha_0, track_seq, track_diagnostics, s, Sigma, beta1, beta2, delta, verb)        
        self.h_tol = h_tol
        self.configure_step_rule_(step_type, local_lipschitz_scale, min_stepsize,
                                  max_stepsize, domain_bt_factor,
                                  domain_bt_max_iters, domain_bt_tol)
        dagness_prev = self.dagness(self.W_est)

        for i in range(iters_out):
            self._fista_restarts_current = 0
            # Minimize augmented Lagrangian to estimate W
            self.W_est, stepsize = self.minimize_primal(self.W_est, self.Sigma_est, lamb, self.alpha, stepsize,
                                                        iters_in, checkpoint, tol, track_seq)

            # Update augmented Lagrangian parameters
            dagness = self.dagness(self.W_est)
            alpha_before = self.alpha
            rho_before = self.rho

            if self.h_tol is not None and dagness <= self.h_tol:
                self.record_diagnostics_(self.W_est, self.Sigma_est, lamb, alpha_before,
                                         rho_before, stepsize, i+1)
                if self.track_diagnostics:
                    self.diagnostics[-1]['stopped_by_h_tol'] = True
                if verb:
                    print(f'- {i+1}/{iters_out}. Diff: {self.diff_W[-1]:.6f} | Acycl: {dagness:.6f}' +
                          f' | Rho: {self.rho:.3f} - Alpha: {self.alpha:.3f} - Step: {stepsize:.4f}' +
                          ' | Stop: h_tol')
                break
            
            # Update Lagrange multiplier
            self.alpha += rho_before*dagness
            self.rho = beta*rho_before if dagness > gamma*dagness_prev else rho_before

            self.record_diagnostics_(self.W_est, self.Sigma_est, lamb, alpha_before,
                                     rho_before, stepsize, i+1)
            if self.track_diagnostics:
                self.diagnostics[-1]['stopped_by_h_tol'] = False

            dagness_prev = dagness

            if dec_step:
                stepsize *= dec_step

            if verb:
                print(f'- {i+1}/{iters_out}. Diff: {self.diff_W[-1]:.6f} | Acycl: {dagness:.6f}' +
                      f' | Rho: {self.rho:.3f} - Alpha: {self.alpha:.3f} - Step: {stepsize:.4f}')
                                    
        return self.W_est  

    def compute_gradient_W_(self, W, Sigma, lamb, alpha):
        G_loss = self.Cx @(W - self.Id) / Sigma / 2 + lamb
        acyc_val = self.dagness(W)
        G_acyc = self.gradient_acyclic(W)
        return G_loss + (alpha + self.rho*acyc_val)*G_acyc

    def compute_gradient_and_stepsize_(self, W, Sigma, lamb, alpha, stepsize):
        self._last_local_lipschitz = np.nan
        if self.acyc_const == 'logdet' and self.uses_local_lipschitz_():
            G_loss = self.Cx @(W - self.Id) / Sigma / 2 + lamb
            h_val, G_acyc, grad_h_norm_sq, _ = self.logdet_terms_(W)
            penalty_weight = alpha + self.rho*h_val
            grad = G_loss + penalty_weight*G_acyc
            lipschitz = (self.loss_lipschitz_(Sigma)
                         + abs(penalty_weight)*grad_h_norm_sq
                         + max(self.rho, 0)*grad_h_norm_sq)
            self._last_local_lipschitz = lipschitz
            stepsize = self.clip_stepsize_(self.local_lipschitz_scale / max(lipschitz, 1e-12))
            return grad, stepsize

        return self.compute_gradient_W_(W, Sigma, lamb, alpha), stepsize

    def init_variables_(self, X, rho_init, alpha_init, track_seq, track_diagnostics, s, Sigma, beta1, beta2, delta, verb):
        super().init_variables_(X, track_seq, track_diagnostics, s, Sigma, beta1, beta2, delta, verb)
        self.rho = rho_init
        self.alpha = alpha_init


class MetMulColide(MetMulDagma):
    """
    Method of ultipliers algorithm for learning DAGs with unknown exogenous covarian using a
    convex version of CoLiDE with DAGMA acyclicity constraint
    """
    def __init__(self, primal_opt='pgd', acyclicity='logdet', restart=False):
        super().__init__(primal_opt, acyclicity, restart)
        if self.opt_type == 'sca':
            self.minimize_primal = self.succ_conv_approx_
    
    def fit(self, X, lamb, stepsize, s=1, iters_in=1000, iters_out=10, checkpoint=250, tol=1e-6,
            beta=5, gamma=.25, rho_0=1, alpha_0=.1, track_seq=False, dec_step=None,
            track_diagnostics=False, beta1=.99, beta2=.999, Sigma=None, scale_sig=.01, delta=.01, sca_adam=False, verb=False):
        
        self.init_variables_(X, rho_0, alpha_0, track_seq, track_diagnostics, s, Sigma, scale_sig, beta1, beta2, delta, verb)        
        
        self.sca_adam = sca_adam

        dagness_prev = self.dagness(self.W_est)
        for i in range(iters_out):
            self._fista_restarts_current = 0
            # Minimize augmented Lagrangian to estimate W
            self.W_est, self.Sigma, stepsize = self.minimize_primal(self.W_est, self.Sigma_est, lamb, self.alpha,
                                                                    stepsize, iters_in, checkpoint, tol, track_seq)

            ##### THIS CODE IS REPEATED ##### 
            # Update augmented Lagrangian parameters
            dagness = self.dagness(self.W_est)
            alpha_before = self.alpha
            rho_before = self.rho
            self.rho = beta*self.rho if dagness > gamma*dagness_prev else self.rho

            # Update Lagrange multiplier
            self.alpha += self.rho*dagness

            self.record_diagnostics_(self.W_est, self.Sigma, lamb, alpha_before,
                                     rho_before, stepsize, i+1)

            dagness_prev = dagness

            if dec_step:
                stepsize *= dec_step
            
            if verb:
                print(f'- {i+1}/{iters_out}. Diff W: {self.diff_W[-1]:.6f} | Diff Sigma: {self.diff_Sig[-1]:.6f}' +
                      f' | Acycl: {dagness:.6f} |Rho: {self.rho:.3f} - Alpha: {self.alpha:.3f} - Step: {stepsize:.4f}')
            ##############################
        return self.W_est, self.Sigma
    
    def init_variables_(self, X, rho_init, alpha_init, track_seq, track_diagnostics, s, Sigma_init, scale_sig, beta1, beta2, delta, verb):
        # Default initialization of Sigma0        
        if Sigma_init is None:
            Sigma_init = np.linalg.norm(X, axis=0) / np.sqrt(X.shape[0])

        super().init_variables_(X, rho_init, alpha_init, track_seq, track_diagnostics, s, Sigma_init, beta1, beta2, delta, verb)
        
        self.Sigma_0 = Sigma_init * scale_sig
        self.Gsig_obj_func = None # for restart

        # For Adam
        self.opt_m_sig, self.opt_v_sig = 0, 0

        # For tracking Sigma sequences
        self.diff_Sig = []
        self.seq_Sig = [] if track_seq else None

    def track_variables_(self, W, W_prev, Sigma, Sigma_prev, track_seq):
        self.track_W_variable_(W, W_prev, track_seq)

        norm_Sig_prev = la.norm(Sigma_prev)
        norm_Sig_prev = norm_Sig_prev if norm_Sig_prev != 0 else 1
        self.diff_Sig.append(la.norm(Sigma - Sigma_prev) / norm_Sig_prev)
        if track_seq:
            self.seq_Sig.append(Sigma)

    def proj_grad_step_Sigma_(self, Sigma, W, stepsize, iter):
        self._last_eval_sigma_min = np.min(Sigma)
        # Compute the gradient
        v_aux = np.diag( (self.Id - W.T) @ self.Cx @ (self.Id - W) )
        self.Gsig_obj_func = -.5 / Sigma * v_aux / Sigma + .5
        if self.opt_type == 'adam':
            self.Gsig_obj_func, self.opt_m_sig, self.opt_v_sig \
                = self.compute_adam_grad_(self.Gsig_obj_func, self.opt_m_sig, self.opt_v_sig, iter+1)
        # Project to feasible set 
        Sigma_est = np.maximum(Sigma - stepsize*self.Gsig_obj_func, self.Sigma_0)
        
        return Sigma_est
    
    # NOTE: try using an adam step instead of a simple gradient step
    def succ_conv_approx_(self, W, Sigma, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                          track_seq):
        """
        Succesive Convex Approximation algorithm that estiamtes W and Sigma via an alternating
        minimization scheme where at each iteration minimizes an upper bound with closed form solution
        """
        W_prev = W.copy()
        Sigma_prev = Sigma.copy()

        self.opt_type = "adam" if self.sca_adam else self.opt_type
        for i in range(max_iters):
            # Closed form solution of upperbound of W (coincides with a single gradient step) 
            W, stepsize = self.proj_grad_step_W_(W_prev, Sigma_prev, alpha, lamb, stepsize, 0)

            # Closed form solucion of Sigma
            v_aux = np.diag( (self.Id - W.T) @ self.Cx @ (self.Id - W) )
            Sigma = np.maximum( np.sqrt(v_aux), self.Sigma_0 )

            # Update tracking variables
            self.track_variables_(W, W_prev, Sigma, Sigma_prev, track_seq)

            # Check convergence
            if i % checkpoint == 0 and (self.diff_W[-1] + self.diff_Sig[-1]) / 2 <= tol:
                break
    
            W_prev = W.copy()
            Sigma_prev = Sigma.copy()

        self._last_inner_iters = i + 1
        return W, Sigma, stepsize
    
    def proj_grad_desc_(self, W, Sigma, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                        track_seq):
        W_prev = W.copy()
        Sigma_prev = Sigma.copy()
        for i in range(max_iters):
            # Gradient step for W
            W, stepsize = self.proj_grad_step_W_(W_prev, Sigma_prev, alpha, lamb, stepsize, i)

            # Gradient step for Sigma
            Sigma = self.proj_grad_step_Sigma_(Sigma_prev, W_prev, stepsize, i)

            # Update tracking variables
            self.track_variables_(W, W_prev, Sigma, Sigma_prev, track_seq)

            # Check convergence
            if i % checkpoint == 0 and (self.diff_W[-1] + self.diff_Sig[-1]) / 2 <= tol:
                if self.verb:
                    print("Convergence achieved at iter", i+1)
                break
    
            W_prev = W.copy()
            Sigma_prev = Sigma.copy()

        self._last_inner_iters = i + 1
        return W, Sigma, stepsize
    
    def acc_proj_grad_desc_(self, W, Sigma, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                            track_seq):
        W_prev, W_fista = W.copy(), W.copy()
        Sigma_prev, Sigma_fista = Sigma.copy(), Sigma.copy()
        t_k = 1
        for i in range(max_iters):
            W, stepsize = self.proj_grad_step_W_(W_fista, Sigma_fista, alpha, lamb, stepsize, i)
            diff_W = W - W_prev

            # Sigma = self.proj_grad_step_Sigma_(Sigma_fista, W_prev, stepsize, i)
            Sigma = self.proj_grad_step_Sigma_(Sigma_fista, W_fista, stepsize, i)
            diff_Sig = Sigma - Sigma_prev
    
            # Update tracking variables
            self.track_variables_(W, W_prev, Sigma, Sigma_prev, track_seq)

            # Check if restarting condition is met
            inner_prod_grad_W = np.vdot(self.Gw_obj_func, diff_W)
            inner_prod_grad_Sig = self.Gsig_obj_func.T @ diff_Sig
            if self.restart and (inner_prod_grad_W + inner_prod_grad_Sig) > 1e-6:
                self._fista_restarts_current += 1
                W = W_prev.copy()      
                W_fista = W.copy()
                Sigma = Sigma_prev.copy()
                Sigma_fista = Sigma.copy()
                t_k = 1
                continue

            t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
            W_fista = W + (t_k - 1)/t_next*(diff_W)
            Sigma_fista = Sigma + (t_k - 1)/t_next*(diff_Sig)

            
            # Check convergence
            if i % checkpoint == 0 and (self.diff_W[-1] + self.diff_Sig[-1]) / 2 <= tol:
                if self.verb:
                    print("Convergence achieved at iter", i+1)
                break

            W_prev = W
            Sigma_prev = Sigma
            t_k = t_next

        self._last_inner_iters = i + 1
        return W, Sigma, stepsize
