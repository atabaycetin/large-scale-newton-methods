function [g] = findiff_grad(f, x, h, type)
    n = length(x);
    g = zeros(n, 1);
    
    % compute f(x)
    if strcmpi(type, 'fw')
        fx = f(x);
    end
    
    % we handle both scalar h and vector h 
    % (this makes the func a good wrapper)
    if length(h) == 1
        h_vec = h * ones(n, 1);
    else
        h_vec = h;
    end
    
    for k = 1:n
        hk = h_vec(k);
        
        % safety against avoid division by zero if step size is 0
        if hk == 0
            hk = 1e-8; 
        end
        
        % in-place modification to avoid memory allocation
        xk_orig = x(k);
        
        if strcmpi(type, 'fw')
            % forward difference formula: (f(x+h) - f(x)) / h
            x(k) = xk_orig + hk;
            g(k) = (f(x) - fx) / hk;
            
        elseif strcmpi(type, 'c')
            % central difference formula: (f(x+h) - f(x-h)) / 2h
            
            % forward point
            x(k) = xk_orig + hk;
            f_plus = f(x);
            
            % backward point
            x(k) = xk_orig - hk;
            f_minus = f(x);
            
            g(k) = (f_plus - f_minus) / (2 * hk);
        else
            error("Type must be either 'fw' or 'c'!");
        end
        
        % restore original x for the next iteration
        x(k) = xk_orig; 
    end
end