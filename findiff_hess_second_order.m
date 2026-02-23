function [H] = findiff_hess_second_order(f, x, h)
    n = length(x);
    H = zeros(n, n);
    
    % handle h vector
    if length(h) == 1
        h_vec = h * ones(n, 1);
    else
        h_vec = h(:);
        if length(h_vec) ~= n
            error("h must be scalar or length(x).");
        end
    end
    
    if any(h_vec == 0)
        error("h contains zeros.");
    end
    
    fx = f(x);
    
    % fix direction
    for i = 1:n
        hi = h_vec(i);

        % H_ii ~= [ f(x + h_i e_i) - 2 f(x) + f(x - h_i e_i) ] / h_i^2
        % [ f(x+hi) - 2f(x) + f(x-hi) ] / hi^2
        
        x_plus = x;   x_plus(i) = x(i) + hi;
        x_minus = x;  x_minus(i) = x(i) - hi;
        
        H(i,i) = (f(x_plus) - 2*fx + f(x_minus)) / (hi^2);
        
        for j = i+1:n
            hj = h_vec(j);

            %   H_ij = H_ji ~= [  f(x + h_i e_i + h_j e_j)
            %                   - f(x + h_i e_i - h_j e_j)
            %                   - f(x - h_i e_i + h_j e_j)
            %                   + f(x - h_i e_i - h_j e_j) ] / (4 h_i h_j)

            % 4 points ++, +-, -+, --
            % x + hi + hj
            x_pp = x; x_pp(i) = x(i) + hi; x_pp(j) = x(j) + hj;
            % x + hi - hj
            x_pm = x; x_pm(i) = x(i) + hi; x_pm(j) = x(j) - hj;
            % x - hi + hj
            x_mp = x; x_mp(i) = x(i) - hi; x_mp(j) = x(j) + hj;
            % x - hi - hj
            x_mm = x; x_mm(i) = x(i) - hi; x_mm(j) = x(j) - hj;
            
            val = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * hi * hj);
            
            H(i,j) = val;
            H(j,i) = val;
        end
    end
    
    % enforce symmetry in case of floating point error
    % bc hessians of smooth functions must be symmetric
    H = (H + H')/2;
end