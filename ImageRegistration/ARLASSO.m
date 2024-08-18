function [xhat, zhat, crossvalidationErrorList, usedlambda1, usedscale] = ARLASSO(A1, y1, A2, y2, A_cv, y_cv, r1, r2, H, W, init)

%     Implements Robust Lasso with known correspondences.
%     A: (m + n, p) and y: (m + n, 1), where N = m + n
%     m: number of known correspondences
%     n: number of unknown correspondences
% 
%     A_cv: (cv, p)  y_cv: (cv, 1)
    
 
    m = size(y1, 1);
    n = size(y2, 1);
    p = H * W;
    N = m + n;
    c1 = 1;
    c2 = 1;
    currLeastError = Inf;
    crossvalidationErrorList = zeros(size(r1, 2), size(r2, 2));

    for i = 1 : size(r1, 2)
        for j = 1 : size(r2, 2)
            fprintf("%d, %d, %d, %d", i, j, size(r1, 2), size(r2, 2))
            lambda1 = r1(1, i);
            scale = r2(1, j);

            clear par;
            par.max_iter = 1000;
            [out, ~, ~] = fista(@(t) c1 * norm(y1 - DCTSubmatrixTimesVector(t(1:p, :), A1, H, W), 2)^2 ...
                                   + c2 * norm(y2 - DCTSubmatrixTimesVector(t(1:p, :), A2, H, W) - t(p + 1:end, :), 2)^2, ...
                                @(t) c1 * [DCTSubmatrixTransposeTimesVector(DCTSubmatrixTimesVector(t(1:p, :), A1, H, W) - y1, A1, H, W); zeros(n, 1)] ...
                                   + c2 * [DCTSubmatrixTransposeTimesVector(DCTSubmatrixTimesVector(t(1:p, :), A2, H, W) + t(p+1:end, :) - y2, A2, H, W); ...
                                      DCTSubmatrixTimesVector(t(1:p, :), A2, H, W) + t(p+1:end, :) - y2], ...
                                @(t)  norm(t(1 : p, :), 1) + scale * norm(t(p + 1 : end, :), 1), ...
                             @(t, a) [prox_l1(t(1 : p, :), a); scale * prox_l1(t(p + 1 : end, :), a)], lambda1, [init; zeros(n, 1)], par);
            x = out(1 : p, :);
            z = out(p + 1 : end, :);
            crossValidationError = sqrt((norm(DCTSubmatrixTimesVector(x, A_cv, H, W) - y_cv, 2) ^ 2) / size(y_cv, 1));
%             disp(DCTSubmatrixTimesVector(x, A1, H, W));
%             disp(y1);
%             disp(c1 * norm(y1 - DCTSubmatrixTimesVector(x, A1, H, W), 2)^2);
%             disp(c2 * norm(y2 - DCTSubmatrixTimesVector(x, A2, H, W) - z, 2)^2)
            crossvalidationErrorList(i, j) = crossValidationError;
            if crossValidationError < currLeastError
                currLeastError = crossValidationError;
                xhat = x;
                zhat = z;
                usedlambda1 = lambda1;
                usedscale = scale;
            end
        end
    end
    fprintf("Error on cross validation set is %s \n", num2str(round(currLeastError, 3)))
end