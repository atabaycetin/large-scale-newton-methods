function [H] = findiff_hess(grad_f, x, h)
    n = length(x);
    H = zeros(n, n);
    
    % compute exact gradient g(x)
    gx = grad_f(x);
    
    % handle both scalar h and vector h
    if length(h) == 1
        h_vec = h * ones(n, 1);
    else
        h_vec = h;
    end
    
    for k = 1:n
        hk = h_vec(k);
        
        % safety for very small x_i
        if hk == 0
            hk = 1e-8;
        end
        
        % in-place modification to avoid memory allocation
        xk_orig = x(k);
        
        % perturb x in the k-th direction
        x(k) = xk_orig + hk;
        
        % evaluate gradient at perturbed point
        g_plus = grad_f(x);
        
        % forward difference on the gradient:
        H(:, k) = (g_plus - gx) / hk;
        
        % restore original x for the next iteration
        x(k) = xk_orig;
    end
    
    % enforce symmetry in case of floating point error
    % bc hessians of smooth functions must be symmetric
    H = (H + H') / 2;
end