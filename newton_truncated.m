function [xk, fk, gradfk_norm, k, xseq, btseq, cg_iters, success] = ...
    newton_truncated(x0, f, gradf, HesVP, kmax, tolgrad, c1, rho, btmax, cgmax)
    
    % Function that performs the Truncated Newton Optimization method
    % with conjugate gradient method and backtracking, simultaneously 
    % utilizing hessian-vector product
    %
    % INPUTS:
    % x0 = n-dimensional column vector;
    %
    % f = function handle that describes a function R^n->R;
    %
    % gradf = function handle that describes the gradient of f;
    %
    % Hessf = function handle that computes the Hessian-vector product (HesVP(x, d));
    %
    % kmax = maximum number of iterations permitted;
    %
    % tolgrad = value used as stopping criterion w.r.t. the norm of the
    % gradient;
    %
    % c1 = ﻿the factor of the Armijo condition that must be a scalar in (0,1);
    %
    % rho = ﻿fixed factor, lesser than 1, used for reducing alpha0;
    %
    % btmax = ﻿maximum number of steps for updating alpha during the
    % backtracking strategy;
    %
    % cgmax = magnitude of conjugate gradient iterations to perform;
    %
    
    % OUTPUTS:
    % xk = the last x computed by the function
    %
    % fk = the value f(xk)
    %
    % gradfk_norm = value of the norm of gradf(xk)
    %
    % k = index of the last iteration performed
    %
    % xseq = n-by-k+1 matrix where the columns are the elements xk of the 
    % sequence
    %
    % btseq = 1-by-k+1 vector where elements are the number of backtracking
    % iterations at each optimization step.
    %
    % cg_iters = store number of conjugate gradient iterations performed
    %
    % success = whether or not the method succeeded to converge

    xk = x0;
    k = 0;
    
    % function handle for the armijo condition
    armijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;

    xseq = zeros(length(x0), kmax);
    xseq(:, 1) = x0;
    btseq = zeros(1, kmax);
    cg_iters = zeros(1, kmax);

    % compute gradient
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    while k < kmax && gradfk_norm >= tolgrad
        % setup linear system
        ck = -gradfk;
        
        % initialization for cg
        z = zeros(length(x0), 1);
        r = ck;
        d = r;
        
        % forcing term (superlinear)
        eta_k = min(0.5, sqrt(gradfk_norm));
        
        % inner loop: conjugate gradient
        pk = ck;
        
        % apply conjugate gradient in cgmax steps
        for j = 1 : cgmax
            
            % exact hessian-vector product
            Bd = HesVP(xk, d);
            
            % standard cg logic
            curvature = d' * Bd;
            
            if curvature <= 0
                if j == 1
                    pk = ck;
                else
                    pk = z;
                end
                break;
            end
            
            alpha_cg = (r' * r) / curvature;
            z_new = z + alpha_cg * d;
            r_new = r - alpha_cg * Bd;
            
            % truncation check
            if norm(r_new) <= eta_k * norm(ck)
                pk = z_new;
                break;
            end
            
            beta_cg = (r_new' * r_new) / (r' * r);
            d_new = r_new + beta_cg * d;
            
            z = z_new;
            r = r_new;
            d = d_new;
            
            % if we hit the max iterations, accept current z
            if j == cgmax
                 pk = z;
            end
        end
        cg_iters(k + 1) = j;

        % line search time
        alpha = 1;
        fk = f(xk);
        
        % pre-compute dot product for armijo
        c1_gradfk_pk = c1 * gradfk' * pk;
        
        bt = 0;
        while bt < btmax && f(xk + alpha * pk) > armijo(fk, alpha, c1_gradfk_pk)
            alpha = rho * alpha;
            if alpha < 1e-12
                warning('Line search failed at iteration %d', k);
                break;
            end
            bt = bt + 1;
        end

        % update variables and append values
        xk = xk + alpha * pk;
        k = k + 1;
        xseq(:, k+1) = xk;
        btseq(k) = bt;
        gradfk = gradf(xk);
        gradfk_norm = norm(gradfk);
    end
    
    if gradfk_norm < tolgrad
        success = 'Success';
    else
        success = 'Failure';
    end
    
    % cut the arrays into right size
    cg_iters = cg_iters(1:k);
    xseq = xseq(:, 1:k+1);
    btseq = btseq(1:k);
end