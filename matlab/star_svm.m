function [predict, ranking, alpha error, F, AUC] = star_svm(K, y, labeled_ind, varargin)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% STAR-SVM is a semi-supervised learning classifier  designed to %%%
    %%% adaptively modify the optimization by adjusting the weights at %%%
    %%% each iteration. Labels at each iteration will be weighted less %%%
    %%% than previous iterations.                                      %%%
    %%% -------------------------------------------------------------- %%%
    %%% Inputs:                                                        %%%
    %%% K                       Kernel Matrix  (n x n)                 %%%
    %%% y                       Binary label vector (+/- 1)^n          %%%
    %%% labeled_ind             Indices that are labeled (n x 1)       %%%
    %%% (accepted by varargin)                                         %%%
    %%% C                       Loss regularization weight (0, inf)    %%%
    %%% gamma                   Confidence weight (0,1)                %%%
    %%% r                       Majority lass proportion (0,1)         %%%
    %%% -------------------------------------------------------------- %%%
    %%% Outputs:                                                       %%%
    %%% predict                 Binary prediction vector (+/-1)^n      %%%
    %%% ranking                 Ranking score used for AUC             %%%
    %%% alpha                   Final computed SVM score               %%%
    %%% [error, f_measure, AUC] Error, F1_score, AUC                   %%%
    %%% -------------------------------------------------------------- %%%
    %%% Written by Edward Cheung (eycheung@uwaterloo.ca) 2015          %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Default params
    N = size(K,1);
    C_orig = 1;
    gamma = 0.7;
    debug = 0;
    r = 0.5;
    max_iter = 1000;
    warm_start = 0;
    
    % Read in Optional Params
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case lower('C'),            C_orig = varargin{i+1};
            case lower('gamma'),        gamma = varargin{i+1};  
            case lower('debug'),        debug = varargin{i+1};
            case lower('class_prop'),   r = varargin{i+1};
            case lower('max_iter'),     max_iter = varargin{i+1};
            case lower('warm_start'),   warm_start = varargin{i+1};
            otherwise,
                error('Unexpected parameter %s\n', varargin{i})
        end
    end    
    
    % Initialize Variables
    orig_unlabeled = setdiff(1:N, labeled_ind);       
    unlabeled_ind = orig_unlabeled;
    C = zeros(N,1);
    predict = zeros(N,1);
    predict(labeled_ind) = y(labeled_ind);
    
    orig_unlabeled_size = length(orig_unlabeled);
    orig_labeled_size = length(labeled_ind);
    ranking = zeros(N,1);
    
    % TWO-SVM Variables
    dW = ones(length(labeled_ind),1);
    alpha = zeros(length(labeled_ind), 1);
    T_twosvm = 0;             
    
    C(labeled_ind) = C_orig;
    % Factor of 2 below is just so that using r = 0.5 will not affect
    % original C.
    C(y == 1) = 2*r*C(y == 1);
    C(y == -1) = 2*(1-r)*C(y == -1);
    C_plus = 2*r*C_orig;
    C_neg = 2*(1-r)*C_orig;
    
    k = 1;    
    
    while ~isempty(unlabeled_ind) && k <= max_iter
        M = K(labeled_ind, labeled_ind).*(predict(labeled_ind)*predict(labeled_ind)');
        if warm_start
            [alpha, T_twosvm, sv_ind, sv_coef] = two_svm( M, C(labeled_ind), 'alpha', alpha, 'T', T_twosvm );
        else
            [alpha, T_twosvm, sv_ind, sv_coef] = two_svm( M, C(labeled_ind) );
        end
        
        % Check T_twosvm vs alpha*K*alpha - sum(alpha) here
        g = K(unlabeled_ind, labeled_ind(sv_ind))*diag(predict(labeled_ind(sv_ind)))*sv_coef;
        T = threshold(g, length(labeled_ind) - orig_labeled_size, orig_unlabeled_size);
        
        for T_ind = 1:length(T)
            j = T(T_ind);
            if g(j) > 0
                C(unlabeled_ind(j)) = (gamma^k)*C_plus;
            else
                C(unlabeled_ind(j)) = (gamma^k)*C_neg;
            end
        end
        
        to_be_labeled = unlabeled_ind(T);
        predict(to_be_labeled) = sign(g(T));        
        ranking(to_be_labeled) = length(unlabeled_ind)*predict(to_be_labeled);
        unlabeled_ind = setdiff(unlabeled_ind, to_be_labeled);
        labeled_ind = union(labeled_ind, to_be_labeled, 'stable');
                
        if debug
           fprintf('Iter #%d: length(unlabeled_ind) = %d\n', k, length(unlabeled_ind)); 
        end
        k = k + 1;                
    end     
    alpha = two_svm( K(labeled_ind, labeled_ind), C(labeled_ind));
    [error, F, AUC] = perf_calc(predict(orig_unlabeled), y(orig_unlabeled), ranking(orig_unlabeled));
end

function T = threshold(g, added_labels, orig_unlabeled_size)
    tau = added_labels/orig_unlabeled_size;
    T_plus = find(g >= (1 - tau)*max(g));
    T_minus = find(-g >= (1 - tau)*max(-g));
    T = union(T_plus, T_minus);
end