function [Hv] = findiff_hesvp(grad_f, x, v, h)
    % if h comes in as a vector, strategy='relaive',
    % we must convert it to a scalar because we are moving in direction v,
    % not along the axes. A safe choice is the norm or the min value.
    if length(h) > 1
        % heuristic: use the mean magnitude of the 
        % relative vector bc why not
        epsilon = mean(h); 
    else
        epsilon = h;
    end
    
    % avoid division by zero
    if epsilon == 0
        epsilon = 1e-8;
    end
    
    % compute exact gradient at x
    gx = grad_f(x);
    
    % perturb x in the direction of v
    x_perturbed = x + epsilon * v;
    g_perturbed = grad_f(x_perturbed);
    
    % approximate finite difference
    Hv = (g_perturbed - gx) / epsilon;
end