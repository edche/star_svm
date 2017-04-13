function K = create_kernel(X, kernel_type, varargin)
    
    sig = 1;
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case lower('sig'),        sig = varargin{i+1};            
            otherwise,
                error('Unexpected parameter %s\n', varargin{i})
        end
    end


    if lower(kernel_type) == 'rbf'
        % Computes an rbf kernel matrix from the input coordinates
        %OUTPUTS
        % K = the rbf kernel matrix ( = exp(-1/(2*sigma^2)*(X*X')^2) )

        %Author: Tijl De Bie, february 2003. Adapted: october 2004 (for speedup).

            n=size(X,1);
            K=X*X'/(2*sig^2);
            d=diag(K);
            K=K-ones(n,1)*d'/2;
            K=K-d*ones(1,n)/2;
            K=exp(K);
    elseif lower(kernel_type) == 'linear'
       K = X*X'; 
    end

end