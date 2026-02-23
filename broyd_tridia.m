function [f, g, H] = broyd_tridia(x)
    % Generalized Broyden Tridiagonal Function
    % returns function value f, gradient g, and sparse Hessian H
    
    n = length(x);
    p = 7/3; 

    x_padded = [0; x; 0]; 
    
    % indices for vectorization
    i = 2:n+1;

    u = (3 - 2*x_padded(i)).*x_padded(i) - x_padded(i-1) - x_padded(i+1) + 1;
    
    % function val
    f = sum(abs(u).^p);
    
    if nargout > 1
        sign_u = sign(u);
        abs_u_p1 = abs(u).^(p-1);
        abs_u_p2 = abs(u).^(p-2);
        
        hu_prime = p * sign_u .* abs_u_p1;
        hu_dprime = p * (p-1) * abs_u_p2;
        
        du_dxi = 3 - 4*x;
        
        % gradient calculation
        hu_prime_prev = [hu_prime(2:end); 0]; 
        hu_prime_next = [0; hu_prime(1:end-1)];
        
        g = hu_prime .* du_dxi - hu_prime_prev - hu_prime_next;
    end
    
    if nargout > 2
        % hessian calculation (sparse tridiagonal)
        term_k = hu_dprime .* (du_dxi.^2) - 4 * hu_prime;
        
        term_prev = [hu_dprime(2:end); 0];
        
        term_next = [0; hu_dprime(1:end-1)];
        
        main_diag = term_k + term_prev + term_next;

        off_diag_1 = -hu_dprime(1:end-1) .* du_dxi(1:end-1);
        
        off_diag_2 = -hu_dprime(2:end) .* du_dxi(2:end);
        
        super_diag = off_diag_1 + off_diag_2;
        
        % construct sparse matrix
        B = [super_diag; 0];
        A = main_diag;
        C = [0; super_diag];
        
        H = spdiags([C A B], -1:1, n, n);
    end
end