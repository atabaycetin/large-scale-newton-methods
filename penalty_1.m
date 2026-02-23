function [f, g, H] = penalty_1(x)
    
    n = length(x);
    a = 1e-5;
    
    % function value:
    % f(x) = sum( a*(xi - 1)^2 ) + ( sum(xi^2) - 0.25 )^2

    term1 = a * sum((x - 1).^2);

    sq_sum = x' * x;
    bracket = sq_sum - 0.25;
    term2 = bracket^2;
    
    f = term1 + term2;
    
    if nargout > 1
        % gradient calculation
        g = 2 * a * (x - 1) + 4 * bracket * x;
    end
    
    if nargout > 2
        % hessian calculation
        % H_ij = d(g_i)/d(x_j)
        % diagonal: 2*a + 4*bracket + 4*x_i * 2*x_i
        % off-diag: 4*x_i * 2*x_j
        % H = diag(2*a + 4*bracket) + 8 * x * x'
        
        % construct dense hessian
        % diag part
        diag_term = (2 * a + 4 * bracket);
        
        % we know penalty_1 Hessian structure: H = d*I + 8*x*x'
        % we exploit the shape of the function and return 
        if length(x) >= 1000
            H.type  = 'Sherman-Morrison';
            H.d     = diag_term;
            H.u     = x;
            H.sigma = 8;
        else
            % rank-1 update part (8 * x * x')
            rank1_term = 8 * (x * x');
            
            H = diag_term * speye(n) + rank1_term;
        end
    end
end