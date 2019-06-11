%% variational Bayesian logistic regression model selection, using vb_logit_*
%
% This script demonstrates how to use the variational bound to compare
% logistic models of different complexity and choose the most adequate
% model for the given dataset. The script models a quadratic, noisy input
% -> output mapping by polynomials of increasing order. It then selects the
% order that yields the highest variational bound - a proxy for the highest
% Bayesian model evidence - which trades off model complexity with the
% model's ability to capture the data.
%
% Copyright (c) 2014, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% set RNG seed to re-produce arXiv figures
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave, rand("state", 1); randn("state", 1); else, rng(41); end


%% settings
D = 3;              % 'true' polynomial order (D-1)
N = 50;             % size of training set
D_LD = 6;           % polynomial order used for LDA fit
Ds = 1:10;          % order to test VB regression on
x_range = [-5 5];

% generate data
w = randn(D, 1);

x = x_range(1) + (x_range(2) - x_range(1)) * rand(N, 1);
x_test = linspace(x_range(1), x_range(2), 300)';
gen_X = @(x, d) bsxfun(@power, x, 0:(d-1));  % (d-1)'th order polynomial
X = gen_X(x, D);
py = 1 ./ (1 + exp(- X * w));
y = 2 * (rand(N, 1) < py) - 1;
py_test = 1 ./ (1 + exp(- gen_X(x_test, D) * w));
y_test = 2 * (rand(length(py_test), 1) < py_test) - 1;


%% perform model selection
Ls = NaN(1, length(Ds));
pred_loss = NaN(length(Ds), 2);
for i = 1:length(Ds)
    warning('off','Bayes:maxIter');  % avoid warnings for overspecified models
    [w, V, invV, ~, ~, Ls(i)] = vb_logit_fit(gen_X(x, Ds(i)), y);
    warning('on','Bayes:maxIter');
    y_pred =  2 * (vb_logit_pred(gen_X(x, Ds(i)), w, V, invV) > 0.5) - 1;
    y_test_pred =  ...
        2 * (vb_logit_pred(gen_X(x_test, Ds(i)), w, V, invV) > 0.5) - 1;
    pred_loss(i, :) = [mean(y_pred ~= y) mean(y_test_pred ~= y_test)];
end
[~, i] = max(Ls);
D_best = Ds(i)


%% predictions for selected model
% variational bayes
X_VB = gen_X(x, D_best);
[w_VB, V_VB, invV_VB] = vb_logit_fit(X_VB, y);
py_VB = vb_logit_pred(X_VB, w_VB, V_VB, invV_VB);
py_test_VB = vb_logit_pred(gen_X(x_test, D_best), w_VB, V_VB, invV_VB);
% Linear Fisher Discriminant Analysis
y1 = y == 1;
X_LD = gen_X(x, D_LD);
w_LD = NaN(D_LD, 1);
w_LD(2:end) = (cov(X_LD(y1, 2:end)) + cov(X_LD(~y1, 2:end))) \ ...
              (mean(X_LD(y1, 2:end))' - mean(X_LD(~y1, 2:end))');
w_LD(1) = - 0.5 * (mean(X_LD(y1, 2:end)) + mean(X_LD(~y1, 2:end))) * w_LD(2:end);
y_LD = 2 * (X_LD * w_LD > 0) - 1;
y_test_LD = 2 * (gen_X(x_test, D_LD) * w_LD > 0) - 1;
% output train & test prediction error
fprintf('Training set MSE, LDA = %f, VB = %f\n', ...
        mean(y_LD ~= y), mean(2 * (py_VB > 0.5) - 1 ~= y));
fprintf('Test     set MSE, LDA = %f, VB = %f\n', ...
        mean(y_test_LD ~= y_test), mean(2 * (py_test_VB > 0.5) - 1 ~= y_test));


%% plot model selection result
f1 = figure;  hold on;
[ax, h1, h2] = plotyy(Ds-1, Ls, Ds-1, pred_loss(:, 1));
set(h1, 'LineStyle', '-', 'LineWidth', 1, 'Color', [0 0 0]);
set(h2, 'LineStyle', '-', 'LineWidth', 1, 'Color', [0.8 0 0]);
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1]);
xlabel('polynomial order');
axes(ax(1));
plot([1 1] * (D-1), ylim, 'k--', 'LineWidth', 0.5);
ylabel('variational bound');
set(gca, 'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
axes(ax(2));
hold on;
h3 = plot(Ds-1, pred_loss(:, 2), '--', 'LineWidth', 1, 'Color', [0.8 0 0]); 
ylabel('0-1 loss');
legend([h1 h2 h3], 'vari. bound', 'train loss', 'test loss');
set(gca, 'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3), 'XTick', []);


%% plot prediction
f2 = figure;  hold on;  xlim(x_range);  ylim([0 1]);
plot(x_test, py_test, 'k-', 'LineWidth', 1);
plot(x_test, py_test_VB, '--', 'Color', [0.8 0 0], 'LineWidth', 1);
plot(x_test, 1 ./ (1 + exp(- gen_X(x_test, D_LD) * w_LD)), '-', ...
     'Color', [0 0 0.8], 'LineWidth', 1)
plot(x(y1), 1 - 0.05 * rand(size(y(y1))), '+', ...
     'MarkerSize', 4, 'Color', [0.2 0.4 0.8]);
plot(x(~y1), 0.05 * rand(size(y(~y1))), 'o', ...
     'MarkerSize', 4, 'Color', [0.8 0.4 0.2]);
legend('p(y=1)', 'VB p(y=1)', 'LDA p(y=1)', 'y=1', 'y=0');
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1],...
    'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('x');
ylabel('p_{true/model}(y=1)');
