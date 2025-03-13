close all;
format long g;
figure(1);
tiledlayout(2, 3);
rng("default");
fixed = imresize(rgb2gray(imread("fixed.png")), [256 256]);
% figure(1), imshow(fixed);

H = size(fixed, 1);
W = size(fixed, 2);
k = 10;
indices_u = randi([0 20], k, 1);
indices_v = randi([0 20], k, 1);
indices_u = [0; indices_u];
indices_v = [0; indices_v];

alpha = 1000;
theta_x = zeros(H * W, 1);
theta_y = zeros(H * W, 1);
for i = 1 : k + 1
    u = indices_u(i, 1);
    v = indices_v(i, 1);
    indexInVector = W * u + v + 1;
    theta_x(indexInVector, 1) = alpha / (u + v + 1);
    theta_y(indexInVector, 1) = alpha / (u + v + 1);
end

u_x = reshape(idct2(reshape(theta_x, H, W)'), H, W, 1);
u_y = reshape(idct2(reshape(theta_y, H, W)'), H, W, 1);
u = cat(3, u_x, u_y);

% moved = imwarp(fixed, u);
% figure(2), imshow(moved);
% legend("moved");
% 
% n_pure = 23;
% manuallyMarkedPoints1 = zeros(n_pure, 2);
% manuallyMarkedPoints2 = zeros(n_pure, 2);
% 
% for i = 22 : n_pure
%     figure(3), imshow(fixed);
%     title("fixed image")
%     [x, y] = ginput(1);                           % [column, row]
%     manuallyMarkedPoints1(i, 1) = x;
%     manuallyMarkedPoints1(i, 2) = y;
%     fixed = insertMarker(fixed, manuallyMarkedPoints1);
%     imshow(fixed)
% 
%     figure(4), imshow(moved);
%     title("moving image")
%     [x, y] = ginput(1);
%     manuallyMarkedPoints2(i, 1) = x;
%     manuallyMarkedPoints2(i, 2) = y;
%     moved = insertMarker(moved, manuallyMarkedPoints2);
%     imshow(moved)
% end

% save('manuallyMarkedPoints1.mat');
% save('manuallyMarkedPoints2.mat');

% note that u is motion (fixed - moved)

% fixed = imresize(rgb2gray(imread("Elephant/frame-1.png")), [256 256]);
% fixed = imresize(rgb2gray(imread("fixed.png")), [256 256]);
moved = imread("moved.png");
H = size(fixed, 1);
W = size(fixed, 2);
nexttile, imshow(fixed), title("fixed");
nexttile, imshow(moved), title("moved");

manuallyMarkedPoints1 = load('manuallyAnnotatedPoints/manuallyMarkedPoints_fixed.mat').manuallyMarkedPoints1;
manuallyMarkedPoints2 = load('manuallyAnnotatedPoints/manuallyMarkedPoints_moved.mat').manuallyMarkedPoints2;
n_pure = size(manuallyMarkedPoints1, 1);
% n_pure = 23

points1 = detectSIFTFeatures(fixed, ContrastThreshold = 0.0133);
points2 = detectSIFTFeatures(moved, ContrastThreshold = 0.0133);
[f1, vpts1] = extractFeatures(fixed, points1);
[f2, vpts2] = extractFeatures(moved, points2);
indexPairs = matchFeatures(f1, f2) ;
matchedPointsSIFT1 = vpts1(indexPairs(:, 1));
matchedPointsSIFT2 = vpts2(indexPairs(:, 2));

points1 = detectSURFFeatures(fixed);
points2 = detectSURFFeatures(moved);
[f1, vpts1] = extractFeatures(fixed, points1);
[f2, vpts2] = extractFeatures(moved, points2);
indexPairs = matchFeatures(f1, f2) ;
matchedPointsSURF1 = vpts1(indexPairs(:, 1));
matchedPointsSURF2 = vpts2(indexPairs(:, 2));

points1 = detectHarrisFeatures(fixed);
points2 = detectHarrisFeatures(moved);
[f1, vpts1] = extractFeatures(fixed, points1);
[f2, vpts2] = extractFeatures(moved, points2);
indexPairs = matchFeatures(f1, f2) ;
matchedPointsHARRIS1 = vpts1(indexPairs(:, 1));
matchedPointsHARRIS2 = vpts2(indexPairs(:, 2));

points1 = detectORBFeatures(fixed);
points2 = detectORBFeatures(moved);
[f1, vpts1] = extractFeatures(fixed, points1);
[f2, vpts2] = extractFeatures(moved, points2);
indexPairs = matchFeatures(f1, f2) ;
matchedPointsORB1 = vpts1(indexPairs(:, 1));
matchedPointsORB2 = vpts2(indexPairs(:, 2));

points1 = detectBRISKFeatures(fixed);
points2 = detectBRISKFeatures(moved);
[f1, vpts1] = extractFeatures(fixed, points1);
[f2, vpts2] = extractFeatures(moved, points2);
indexPairs = matchFeatures(f1, f2) ;
matchedPointsBRISK1 = vpts1(indexPairs(:, 1));
matchedPointsBRISK2 = vpts2(indexPairs(:, 2));

points1 = detectKAZEFeatures(fixed);
points2 = detectKAZEFeatures(moved);
[f1, vpts1] = extractFeatures(fixed, points1);
[f2, vpts2] = extractFeatures(moved, points2);
indexPairs = matchFeatures(f1, f2) ;
matchedPointsKAZE1 = vpts1(indexPairs(:, 1));
matchedPointsKAZE2 = vpts2(indexPairs(:, 2));

points1 = detectFASTFeatures(fixed);
points2 = detectFASTFeatures(moved);
[f1, vpts1] = extractFeatures(fixed, points1);
[f2, vpts2] = extractFeatures(moved, points2);
indexPairs = matchFeatures(f1, f2) ;
matchedPointsFAST1 = vpts1(indexPairs(:, 1));
matchedPointsFAST2 = vpts2(indexPairs(:, 2));

points1 = detectMinEigenFeatures(fixed);
points2 = detectMinEigenFeatures(moved);
[f1, vpts1] = extractFeatures(fixed, points1);
[f2, vpts2] = extractFeatures(moved, points2);
indexPairs = matchFeatures(f1, f2) ;
matchedPointsMinEigen1 = vpts1(indexPairs(:, 1));
matchedPointsMinEigen2 = vpts2(indexPairs(:, 2));


% matchedPoints1 = cat(1, matchedPointsSIFT1.Location, matchedPointsSURF1.Location,...
%                         matchedPointsHARRIS1.Location, matchedPointsORB1.Location,...
%                         matchedPointsBRISK1.Location, matchedPointsKAZE1.Location,...
%                         matchedPointsFAST1.Location, matchedPointsMinEigen1.Location);
% 
% matchedPoints2 = cat(1, matchedPointsSIFT2.Location, matchedPointsSURF2.Location,...
%                         matchedPointsHARRIS2.Location, matchedPointsORB2.Location,...
%                         matchedPointsBRISK2.Location, matchedPointsKAZE2.Location,...
%                         matchedPointsFAST2.Location, matchedPointsMinEigen2.Location);

matchedPoints1 = cat(1, matchedPointsSIFT1.Location, matchedPointsHARRIS1.Location, matchedPointsSURF1.Location);
matchedPoints2 = cat(1, matchedPointsSIFT2.Location, matchedPointsHARRIS2.Location, matchedPointsSURF2.Location);                

nexttile, showMatchedFeatures(fixed, moved, manuallyMarkedPoints1, manuallyMarkedPoints2);
title("manually matched point-pairs");
nexttile, showMatchedFeatures(fixed, moved, matchedPoints1, matchedPoints2);
title("algo matched point-pairs");

p = H * W;
n_cv = 5;
m = n_pure - n_cv;
n = size(matchedPoints1, 1);
N = m + n;
% n_pure = n_cv + m

fprintf("Total number of matched points: %d", n);

u_x = zeros(n_pure + n, 1);
u_y = zeros(n_pure + n, 1);
correspondingRowsOfDCTMatrix = 1 : 1 : n_pure + n;

% for each manual match
for i = 1 : n_pure

    c1 = manuallyMarkedPoints1(i, 1);   % [r1, c1] --> [r2, c2]
    r1 = manuallyMarkedPoints1(i, 2);
 
    c2 = manuallyMarkedPoints2(i, 1);
    r2 = manuallyMarkedPoints2(i, 2);

    u_x(i, 1) = c1 - c2;
    u_y(i, 1) = r1 - r2;
    correspondingRowsOfDCTMatrix(i) = round(W * r1 + c1 + 1);
end

% for each algorithm match
for i = 1 : n

    c1 = matchedPoints1(i, 1);   % [r1, c1] --> [r2, c2]
    r1 = matchedPoints1(i, 2);
 
    c2 = matchedPoints2(i, 1);
    r2 = matchedPoints2(i, 2);

    u_x(n_pure + i, 1) = c1 - c2;
    u_y(n_pure + i, 1) = r1 - r2;
    correspondingRowsOfDCTMatrix(n_pure + i) = round(W * r1 + c1 + 1);
end

% AR-LASSO approach
% [theta_x_hat, z_x_hat, cv_errors_x, used_lambda_x, used_scale_x] = ARLASSO(correspondingRowsOfDCTMatrix(n_cv + 1 : n_cv + m), u_x(n_cv + 1 : n_cv + m),...
%                  correspondingRowsOfDCTMatrix(n_cv + m + 1 : n_cv + m + n), u_x(n_cv + m + 1 : n_cv + m + n),...
%                  correspondingRowsOfDCTMatrix(1 : n_cv), u_x(1 : n_cv),...
%                  0.2, 1, H, W, zeros(p, 1));  
%  
% [theta_y_hat, z_y_hat, cv_errors_y, used_lambda_y, used_scale_y] = ARLASSO(correspondingRowsOfDCTMatrix(n_cv + 1 : n_cv + m), u_y(n_cv + 1 : n_cv + m),...
%                  correspondingRowsOfDCTMatrix(n_cv + m + 1 : n_cv + m + n), u_y(n_cv + m + 1 : n_cv + m + n),...
%                  correspondingRowsOfDCTMatrix(1 : n_cv), u_y(1 : n_cv),...  
%                  0.1, 0.2, H, W, zeros(p, 1));    


% R-LASSO approach 
[theta_x_hat, z_x_hat, cv_errors_x, used_lambda_x, used_scale_x] = RLASSO(...
                 correspondingRowsOfDCTMatrix(n_cv + 1 : n_cv + m + n), u_x(n_cv + 1 : n_cv + m + n),...
                 correspondingRowsOfDCTMatrix(1 : n_cv), u_x(1 : n_cv),...
                 0.5, 0.9, H, W, zeros(p, 1));  
 
[theta_y_hat, z_y_hat, cv_errors_y, used_lambda_y, used_scale_y] = RLASSO(...
                 correspondingRowsOfDCTMatrix(n_cv + 1 : n_cv + m + n), u_y(n_cv + 1 : n_cv + m + n),...
                 correspondingRowsOfDCTMatrix(1 : n_cv), u_y(1 : n_cv),...  
                 0.1, 0.2, H, W, zeros(p, 1)); 

u_x_hat = reshape(idct2(reshape(theta_x_hat, H, W)'), H, W, 1);
u_y_hat = reshape(idct2(reshape(theta_y_hat, H, W)'), H, W, 1);
u_hat = cat(3, u_x_hat, u_y_hat);
reconstructedMoved = imwarp(fixed, u_hat);

nexttile, imshow(reconstructedMoved), title("reconstructed moved");
nexttile, imshow(imfuse(reconstructedMoved, moved)), title("reconstructed moved and original moved");

figure(2);
tiledlayout(2, 2);
nexttile, stem(theta_x_hat), title("theta_x");
nexttile, stem(z_x_hat), title("z\_x\_hat");
nexttile, stem(theta_y_hat), title("theta_y");
nexttile, stem(z_y_hat), title("z\_y\_hat");

imageReconstructionError = meanNormalizedError(moved, reconstructedMoved)
motionFieldReconstructionError = meanNormalizedError(u, u_hat)



 

