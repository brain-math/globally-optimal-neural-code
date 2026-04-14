function z = r2score(y_true, y_pred)
% R^2 (coefficient of determination)
%
% Inputs:
%   y_true: Ground truth. Shape: n_samples * n_outputs.
%   y_pred: Estimated values. Shape: n_samples * n_outputs.
%
% Output:
%   z: The R^2 score.

numerator = sum((y_true - y_pred).^2);
denominator = sum((y_true - mean(y_true)).^2);
valid_score = denominator ~= 0;
output_scores = ones(size(y_true, 2), 1);
    
output_scores(valid_score) = 1 - numerator(valid_score) ./ denominator(valid_score);
z = mean(output_scores);
