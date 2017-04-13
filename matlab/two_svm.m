function [alpha, T, sv_ind, sv_coef] = two_svm( K, C, varargin )
%TWO_SVM Summary of this function goes here
%   Detailed explanation goes here    

% TODO: Get warm start working
    
    % Initialize Variables
    T = 0;
    n = length(C);    
    dW = ones(n,1);
    S = sum(C);   
    
    alpha = zeros(n,1);
    epsilon = 1E-6;
    max_iter = 1000;
    debug = 0;
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case lower('alpha'),        alpha = varargin{i+1};
            case lower('t'),            T = varargin{i+1};
            case lower('epsilon'),      epsilon = varargin{i+1};            
            case lower('max_iter'),     max_iter = varargin{i+1};            
            case lower('debug'),        debug = varargin{i+1};
            otherwise,
                error('Unexpected parameter %s\n', varargin{i})
        end
    end
    
    % Warm Start Conditions
    E = 0;
    if norm(alpha) ~= 0
        old_size = length(alpha);
        if old_size <= n
            alpha = [alpha(:); zeros(n-old_size,1)];
            dW = ones(n,1) - K*alpha;
            E = sum(C.*(clip(dW, 0, 2)));            
        end        
        S = T + E;
    end
    
    iter = 0;
    % Algorithm 1 1D-SVM solver in Paper
    while S > epsilon && iter < max_iter
        i_star = find_best_ind(dW, alpha, C);
        
        delta = clip(dW(i_star) + alpha(i_star), 0, C(i_star)) - alpha(i_star);
        alpha(i_star) = clip(dW(i_star) + alpha(i_star), 0, C(i_star));
        % Procedure 2 in Paper.
        % Update dW in direction i_star by delta and calculate S
        T = T - delta*(2*dW(i_star) - 1 - delta);
        dW = dW - delta*K(:, i_star);
        E = sum(C.*(clip(dW, 0, 2)));
        S = T + E;
        iter = iter + 1;
        if debug
           fprintf('Iter #%d: S = %f, T = %f, E = %f, alpha[i*] = %f, delta = %f\n', ...
               iter, S, T, E, alpha(i_star), delta);
        end
    end
    
    sv_ind = find(alpha ~= 0);
    sv_coef = alpha(sv_ind);    
end

function i_star = find_best_ind(dW, alpha, C)
    alpha_star = clip(dW + alpha, 0, C);
    delta = alpha_star - alpha;
    gain = delta.*(dW - delta/2);
    i_star = find(gain == max(gain),1);    
end

function x = clip(x, lb, ub)
    % Ensure that lb < ub
    ub = max(lb, ub);
    lb = min(lb, ub);
    clip_below = x < lb;
    clip_above = x > ub;
    in_range = not(clip_below | clip_above);

    x = x.*in_range + lb.*clip_below + ub.*clip_above;    
end
