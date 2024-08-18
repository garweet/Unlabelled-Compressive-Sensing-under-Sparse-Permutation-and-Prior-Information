function res = DCTSubmatrixTransposeTimesVector(u, rowIndices, H, W)

% Computes H(rowIndices, :) * u

    aux_u = zeros(H * W, 1);
    aux_u(rowIndices, :) = u;

%     disp(size(u))
%     disp(rowIndices)

    res = reshape(dct2(reshape(aux_u, H, W)')', H * W, 1);

end
