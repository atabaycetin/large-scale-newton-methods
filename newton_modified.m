function [xk, fk, gradfk_norm, k, xseq, btseq, success] = ...
    newton_modified(x0, f, gradf, Hessf, ...
    kmax, tolgrad, c1, rho, btmax, diag_shift, dsmax)

    % Function that performs the Modified Newton Optimization method, 
    % using diagonal shift method with backtracking
    % 
    % INPUTS:
    % x0 = n-dimensional column vector;
    %
    % f = function handle that describes a function R^n->R;
    %
    % gradf = function handle that describes the gradient of f;
    %
    % Hessf = function handle that describes the Hessian of f;
    %
    % kmax = maximum number of iterations permitted;
    %
    % tolgrad = value used as stopping criterion w.r.t. the norm of the
    % gradient
    %
    % c1 = ﻿the factor of the Armijo condition that must be a scalar in (0,1);
    %
    % rho = ﻿fixed factor, lesser than 1, used for reducing alpha0;
    %
    % btmax = ﻿maximum number of steps for updating alpha during the 
    % backtracking strategy.
    %
    % diag_shift = magnitude of diagonal shift to perform
    %
    % dsmax = maximum iterations of diagonal shift to perform
    % before falling back to steepest descent
    
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
    % success = whether or not the method succeeded to converge

    
    % Function handle for the armijo condition
    armijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;
    
    % Initializations
    xseq = zeros(length(x0), kmax);
    btseq = zeros(1, kmax);
    
    xk = x0;
    fk = f(xk);
    gradfk = gradf(xk);
    k = 0;
    gradfk_norm = norm(gradfk);
    
    while k < kmax && gradfk_norm >= tolgrad

        %------  SOLVING WITH ITERATIVE CHOLESKY METHOD ------%
        %--------------- USING DIAGONAL SHIFT ----------------%

        % evaluate hessian
        H = Hessf(xk);

        % % uncomment below to see properties of the hessian
        % fprintf('Name: %s  Bytes: %d  Size: %dx%d  Class: %s  Is sparse: %d\n', ...
        %     info.name, info.bytes, info.size(1), info.size(2), info.class, info.sparse);
        
        if isstruct(H)
            % POLYMORPHISM STRAT
            % we use polymorphism to handle cases where the hessian
            % would not fit into our ram. In that case, the test
            % function must send a special struct. You can read more
            % in penalty function description. We basically exploit the
            % function structure
            % pull parameters from hessian struct
            d = H.d;
            u = H.u;
            sigma = H.sigma;
        
            % initial diagonal shift setup (ensure base diagonal is positive)
            min_d = min(d);
            tau = 0;
            if min_d <= 0
                tau = -min_d + diag_shift;
            end
            
            % The gradient failed to converge when tau was at least 0.1% 
            % of the average magnitude as indicated below. Hence the part
            % below is not used. You can uncomment it if you want to try

            % since we don't have an H to calculate the frobenious norm of
            % we use the estimate of the H
            % tau = max(tau, 1e-3 * norm(d) + sigma * (u' * u));
            
            % assume failure initially to enter/control the loop
            flag = 1;

            % iterative loop (mimicking the cholesky loop on Sherman-Morrison)
            for t = 1:dsmax
                
                d_eff = d + tau; % This is our "A" matrix (diagonal)
                
                % compute terms needed for the denominator check
                % A_inv * u = u ./ d_eff
                invA_u = u ./ d_eff;
                
                % compute Sherman-Morrison denominator: 1 + sigma * u' * A^-1 * u
                denom = 1 + sigma * (u' * invA_u);
                
                % we check if the matrix is positive definite
                % for this structure, it is PD if d_eff > 0 (guaranteed by tau logic)
                % and denom > 0
                if denom > 1e-12
                    flag = 0;
                    break;
                else
                    % if matrix not PD, increase tau.
                    if tau == 0
                        tau = diag_shift;
                    else
                        tau = max(2 * tau, diag_shift);
                    end
                end
            end
            
            % compute search direction
            if flag == 0
                % re-compute invA_g using the successful d_eff
                invA_g = -gradfk ./ d_eff;
                
                % formula: A_inv*g - [ (sigma * u' * A_inv * g) / denom ] * A_inv * u
                scale_factor = (sigma * (u' * invA_g)) / denom;
                pk = invA_g - scale_factor * invA_u;
            else
                % if we exhausted dsmax without making it PD
                warning('Sherman-Morrison Tau loop failed. Falling back to Steepest Descent.');
                pk = -gradfk;
            end
        else
            % CHOLESKY WITH DIAGONAL SHIFT Strat
            % We look for Bk = H + Ek such that Bk is Positive Definite.
            
            % Attempt standard Cholesky
            [R, flag] = chol(H);
            
            % if it fails
            if flag ~= 0
                % heuristic to find a starting tau:
                % it should be slightly larger than the magnitude of the most negative eigenvalue.
                % since eigenvalues are expensive, we use the minimum diagonal element.
                min_diag = min(diag(H));
                
                if min_diag > 0
                    tau = diag_shift; % Fallback small shift
                else
                    tau = -min_diag + diag_shift; % Ensure diagonal is positive
                end
                
                % if off-diagonals are huge, 1e-6 is too small.
                % ensure tau is at least 0.1% of the average magnitude if the
                % previous heuristic resulted in a tiny tau.
                % (using Frobenious norm is cheap).
                tau = max(tau, 1e-3 * norm(H, 'fro'));
                
                % iteratively increase tau until Cholesky succeeds
                % (this loop usually runs only once or twice)
                for t=1:dsmax
                    [R, flag] = chol(H + tau * speye(length(xk)));
                    if flag == 0
                        break;
                    else
                        % increase tau (e.g., multiply by 2 or 10)
                        tau = max(2 * tau, diag_shift);
                    end
                end
            end
            
            % compute descent direction solving Bk * pk = -gradfk
            % using the Cholesky factor R (where Bk = R' * R)
            if flag == 0
                y = R' \ (-gradfk);
                pk = R \ y;
            else
                warning('Cholesky failed! Falling back to Steepest Descent')
                pk = -gradfk;
            end
        end
        
        % % Following check is not necessary unless matrix is
        % % too ill-conditioned or there are precision errors
        % if gradfk' * pk >= 0
        %     warning('pk is not a descent direction! Falling back to Steepest Descent')
        %     pk = -gradfk;
        % end
        
        % reset the value of alpha
        alpha = 1;
        
        % compute the candidate new xk
        xnew = xk + alpha * pk;
        % compute the value of f in the candidate new xk
        fnew = f(xnew);
        
        c1_gradfk_pk = c1 * gradfk' * pk;
        bt = 0;
        % Backtracking strategy: 
        % 2nd condition is the Armijo condition not satisfied
        while bt < btmax && fnew > armijo(fk, alpha, c1_gradfk_pk)
            % Reduce the value of alpha
            alpha = rho * alpha;
            % Update xnew and fnew w.r.t. the reduced alpha
            xnew = xk + alpha * pk;
            fnew = f(xnew);
            
            % Increase the counter by one
            bt = bt + 1;
        end
        if bt == btmax && fnew > armijo(fk, alpha, c1_gradfk_pk)
            warning('newton:linesearch', 'Line search failed to satisfy Armijo at k = %d. Stopping.', k+1);
            break
        end
        
        % Update xk, fk, gradfk_norm
        xk = xnew;
        fk = fnew;
        gradfk = gradf(xk);
        gradfk_norm = norm(gradfk);
        
        % Increase the step by one
        k = k + 1;
        
        % Store current xk in xseq
        xseq(:, k) = xk;

        % Store bt iterations in btseq
        btseq(k) = bt;
    end


    if gradfk_norm < tolgrad
        success = 'Success';
    else
        success = 'Failure';
    end
    
    % cut xseq and btseq to the right size
    xseq = xseq(:, 1:k);
    xseq = [x0, xseq];
    btseq = btseq(1:k);
end