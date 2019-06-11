%% model selection with vb_linear_*
%
% This script demonstrates how to use the variational bound to compare
% linear models of different complexity and choose the most adequate model
% for the given dataset. The script models a quadratic, noisy input ->
% output mapping by polynomials of increasing order. It then selects the
% order that yields the highest variational bound - a proxy for the highest
% Bayesian model evidence - which trades off model complexity with the
% model's ability to capture the data.
%
% Copyright (c) 2014, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% set RNG seed to re-produce arXiv figures (in MATLAB)
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave, rand("state", 0); randn("state", 0); else, rng(1); end


%% settings
D = 3;              % 'true' polynomial order (D-1)
N = 10;             % size of training set
D_ML = 6;           % polynomial order used for maximum likelihood fit
Ds = 1:10;          % orders to test for VB regression
x_range = [-5 5];

% generate data
w = randn(D, 1);
x = x_range(1) + (x_range(2) - x_range(1)) * rand(N, 1);
x_test = linspace(x_range(1), x_range(2), 100)';
gen_X = @(x, d) bsxfun(@power, x, 0:(d-1));  % (d-1)'th order polynomial
X = gen_X(x, D);
% output is based on (D-1)'th order polynomial
y = X * w + randn(N, 1);
y_test = gen_X(x_test, D) * w;


%% perform model selection
% this is done by fitting polynomials of different order, and comparing their
% associated variational bound. The model with the highest bound is deemed to
% feature the best fit.
Ls = NaN(1, length(Ds));
for i = 1:length(Ds)
    [~, ~, ~, ~, ~, ~, ~, Ls(i)] = vb_linear_fit(gen_X(x, Ds(i)), y);
end
[~, i] = max(Ls);
D_best = Ds(i);


%% predictions for selected model
% variational bayes, predictions for training & test set
[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(gen_X(x, D_best), y);
[y_VB, lam_VB, nu_VB] = ...
    vb_linear_pred(gen_X(x_test, D_best), w_VB, V_VB, an_VB, bn_VB);
y_VB_sd = sqrt(nu_VB ./ (lam_VB .* (nu_VB - 2)));
% maximum likelihood, predictions for training & test set
w_ML = gen_X(x, D_ML) \ y;
y_ML = gen_X(x_test, D_ML) * w_ML;
% output test set prediction error
fprintf('Test set MSE, ML = %f, VB = %f\n', ...
        mean((y_test - y_ML).^2), mean((y_test - y_VB).^2));


%% plot model selection result
f1 = figure;  hold on;
plot(Ds-1, Ls, 'k-', 'LineWidth', 1);
plot([1 1] * (D - 1), ylim, 'k--', 'LineWidth', 0.5);
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
    'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('polynomial order');
ylabel('variational bound');


%% plot prediction
f2 = figure;  hold on;
% shaded CI area
patch([x_test; flipud(x_test)], ...
      [(y_VB + 1.96 * y_VB_sd); flipud(y_VB - 1.96 * y_VB_sd)], ...
      [1 1 1] * 0.9, 'EdgeColor', 'none');
% true and esimtated outputs
h1 = plot(x_test, y_test, 'k-', 'LineWidth', 1);
h2 = plot(x_test, y_VB, '--', 'Color', [0.8 0 0], 'LineWidth', 1);
h3 = plot(x_test, y_ML, '-.', 'Color', [0 0 0.8], 'LineWidth', 1);
h4 = plot(x, y, 'k+', 'MarkerSize', 5);
legend([h1 h2 h3 h4], {'true', 'VB', 'ML', 'data'});
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
    'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('x');
ylabel('y, y_{ML}, y_{VB}');
