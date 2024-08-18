function temp = DCTSubmatrixTimesVector(u, rowIndices, H, W)

% Computes H(rowIndices, :) * u
    temp = idct2(reshape(u, H, W)');
    temp = reshape(temp', H * W, 1);

%     disp(rowIndices)
%     disp(size(res))
    temp = temp(rowIndices, :);

end
