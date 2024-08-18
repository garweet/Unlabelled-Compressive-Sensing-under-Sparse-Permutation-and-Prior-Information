function error = meanNormalizedError(x, xhat)

%     x, xhat: (p, 1)

    x = double(x);
    xhat = double(xhat);
    error = norm(x - xhat, "fro")/norm(x, "fro");
end
