function [error, F, AUC] = perf_calc(predict, truth, ranking)
    % Function to help calculate key performance metrics

    % Calculate error
    error = sum(predict ~= truth)/length(truth);
    
    % Calculate F1 score
    tp = sum(predict == 1 & truth == 1);
    fp = sum(predict == 1 & truth == -1);
    fn = sum(predict == -1 & truth == 1);
    
    if tp + fn == 0 || tp + fp == 0
        F = 0;
    else
        p = tp/(tp + fp);
        r = tp/(tp + fn);

        F = (2*p*r)/(p + r);
    end
    
    % Calculate AUC
    [~,~,~,AUC] = perfcurve(truth, ranking, 1);
end