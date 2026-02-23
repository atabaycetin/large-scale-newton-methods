clc; clear; close all;

SEED = 348306; % we gotta put our actual min matricola here
n = [2 1e3 1e4 1e5];

rng(SEED);

%% We implemented a switch in the code to handle three cases:
%    Mode,    Gradient,         Hessian,        Step Size (h)
% 1. Exact,   Exact Formula,    Exact Formula,  N/A
% 2. Mixed,   Exact Formula,    FD Approx,      "10−4,10−8,10−12"
% 3. Full FD, FD Approx,        FD Approx,      "10−4,10−8,10−12"

% We implemented almost everything as a modular structure for ease of use
% and clarity.

%% Define function handles and modes
f_handle = @(x) get_f(x);

h_values = [1e-4 1e-8 1e-12];
strategies = {'Constant', 'Relative'};

% % broyden tridiagonal function
% function y = get_f(x)
%     y = broyd_tridia(x);
% end
% 
% % generalized broyden tridiagonal function exact hessian-vector product
% hvp_handle = @(x, v) broyd_tridia_hesvp(x, v);

% penalty 1 function
function y = get_f(x)
    y = penalty_1(x);
end

% penalty 1 function exact hessian-vector producta
hvp_handle = @(x, v) penalty_1_hesvp(x, v);

%% set hyperparameters (you can read their descriptions in function files)
% max iteration
kmax_m = 1000;      kmax_tn = 1000;

% gradient tolerance
tolgrad_m = 1e-4;   tolgrad_tn = 1e-4;

% line search & armijo parameters
c1_m = 1e-4;        c1_tn = 1e-4;
rho_m = 0.3;        rho_tn = 0.75;
btmax_m = 50;       btmax_tn = 50;

% diagonal shift parameters (modified newton)
diag_shift = 1e-6;  dsmax = 50;

% conjugate gradient parameter (truncated newton)
cgmax = 100;

for dim = n
    for h = h_values
        for s = 1:length(strategies)
            % define strategies
            strat = strategies{s};

            % construct hessian handle based on strat
            if strcmp(strat, 'Constant')
                fd_g_handle = @(x) findiff_grad(f_handle, x, h, 'c');
                fd_h_handle = @(x) findiff_hess_second_order(f_handle, x, h);
                fd_hvp_handle = @(x, v) findiff_hesvp(fd_g_handle, x, v, h);
                strat_name = sprintf('Constant h = 10^%d', log10(h));
                
            elseif strcmp(strat, 'Relative')
                fd_g_handle = @(x) findiff_grad(f_handle, x, h * abs(x), 'c');
                fd_h_handle = @(x) findiff_hess_second_order(f_handle, x, h * abs(x));
                fd_hvp_handle = @(x, v) findiff_hesvp(fd_g_handle, x, v, h * norm(x));
                strat_name = sprintf('Relative h = 10^%d * |xi|', log10(h));
            end

            %% Testing with standard starting point X
    
            % % generalized broyden tridiagonal function standard starting point
            % x_start_standard = -1 * ones(dim, 1);
            
            % penalty function 1 standard starting point
            x_start_standard = 1 * ones(dim, 1); % we chose l = 1
            
            fprintf('\n%s\n', repmat('-', 1, 60));
            fprintf('RUNNING FOR n = %d\n', dim);
            fprintf('Strategy: %s\n', strat_name);
        
            fprintf('%s\n', repmat('-', 1, 60));
            
            fprintf('\n--- Method 1: Modified Newton (Cholesky) ---\n');
            
            t_start = tic; % calculate execution time
            [xk, fk, gradfk_norm, k, xseq, btseq, success] = ...
                    newton_modified(x_start_standard, f_handle, fd_g_handle, fd_h_handle, ...
                                    kmax_m, tolgrad_m, c1_m, rho_m, btmax_m, diag_shift, dsmax);
            elapsed_time = toc(t_start);
        
            % Print results of Modified Newton Method
            parse_output(0, k, xk, fk, gradfk_norm, xseq, btseq, 0, success, elapsed_time, 'modified_newton');
        
        
            fprintf('\n--- Method 2: Truncated Newton (CG) ---\n');
            
            t_start = tic; % calculate execution time

            [xk, fk, gradfk_norm, k, xseq, btseq, cg_iters, success] = ...
                    newton_truncated(x_start_standard, f_handle, fd_g_handle, fd_hvp_handle, ...
                                     kmax_tn, tolgrad_tn, c1_tn, rho_tn, btmax_tn, cgmax);
            elapsed_time = toc(t_start);
            
            % Print results of Truncated Newton Method
            parse_output(0, k, xk, fk, gradfk_norm, xseq, btseq, cg_iters, success, elapsed_time, 'truncated_newton');
            
            fprintf('%s\n', repmat('-', 1, 60));
            
            %% Testing with random starting point X
            for i = 1:5
                % initialize random point
                rand_point = x_start_standard(:) + 2 * rand(dim, 1) - 1;
                fprintf('\n--- Method 1: Modified Newton (Cholesky) ---\n');
        
                t_start = tic; % calculate execution time
                [xk, fk, gradfk_norm, k, xseq, btseq, success] = ...
                        newton_modified(rand_point, f_handle, fd_g_handle, fd_h_handle, ...
                                        kmax_m, tolgrad_m, c1_m, rho_m, btmax_m, diag_shift, dsmax);
                                      % kmax, tolgrad, c1, rho, btmax, n_tau_iters
                elapsed_time = toc(t_start);
                
                % Print results of Modified Newton Method
                parse_output(i, k, xk, fk, gradfk_norm, xseq, btseq, 0, success, elapsed_time, 'modified_newton');
                
        
                fprintf('\n--- Method 2: Truncated Newton (CG) ---\n');
                
                t_start = tic; % calculate execution time
                [xk, fk, gradfk_norm, k, xseq, btseq, cg_iters, success] = ...
                        newton_truncated(rand_point, f_handle, fd_g_handle, fd_hvp_handle, ...
                                         kmax_tn, tolgrad_tn, c1_tn, rho_tn, btmax_tn, cgmax);
                elapsed_time = toc(t_start);
                
                % Print results of Truncated Newton Method
                parse_output(i, k, xk, fk, gradfk_norm, xseq, btseq, cg_iters, success, elapsed_time, 'truncated_newton');
        
                fprintf('\n%s\n', repmat('-', 1, 60));
            end
        end
    end
end