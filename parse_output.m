function [] = parse_output(idx, k, xk, fk, gradfk_norm, ...
    xseq, btseq, cg_iters, success, elapsed_time, method)

    % This is just a wrapper function for printing 
    % the outputs of the methods

    if idx == 0
        fprintf('Starting with the standard starting point of the test function\n');
    else
        fprintf('Starting with the randomly created starting point: %s\n', sprintf('x_random_%d', idx));
    end
    fprintf('xk first 2:   %.4e  %.4e\n', xk(1:2));
    fprintf('xk last  2:   %.4e  %.4e\n', xk(end-1:end));
    fprintf('Status:       %s\n', success);
    fprintf('Converged in: %d iterations\n', k);
    fprintf('Conv. Rate:   %.2e\n', calc_conv_rate(xseq));
    fprintf('Final f(x):   %e\n', fk);
    fprintf('Final ||g||:  %e\n', gradfk_norm);
    fprintf('Total BT:     %d\n', sum(btseq));
    if strcmpi(method, 'truncated_newton')
        fprintf('Avg CG Iters: %.1f per step\n', mean(cg_iters));
    end

    fprintf('Execution Time: %.4e\n', elapsed_time);
end