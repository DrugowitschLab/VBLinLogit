%% estimating coefficients and separating hyperplane, using vb_logit_*
%
% This script demonstrates the use of variational Bayesian logistic
% regression applied to a dataset with low-dimensional inputs, to recover
% the regression coefficients, and to generate test-set predictions. Its
% performance is compared to linear disciminant analysis.
%
% Copyright (c) 2014-2019, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% set RNG seed and plot limits to re-produce arXiv figures (in MATLAB)
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave, rand("state", 1); randn("state", 1); else, rng(0); end
wlims = [-2.5 2.5];


%% settings
D = 3;           % dimensionality of input
N = 100;         % size of training set
N_test = 1000;   % size of test set

% generate data
w = randn(D, 1);
X_scale = 5;
% ensure class balance, with roughly 50% of samples obeying
% w1 + w2 x2 + w3 x3 > 0
X = [ones(N, 1) (X_scale*(rand(N, 1)-0.5))];
X = [X (X_scale*(rand(N, 1)-0.5) - (w(1) + X(:,2) * w(2)) / w(3))];
X_test = [ones(N_test, 1) (X_scale*(rand(N_test, 1)-0.5))];
X_test = [X_test (X_scale*(rand(N_test, 1)-0.5) - (w(1) + X_test(:,2) * w(2)) / w(3))];
% p(y)'s, and noisy y's
py = 1 ./ (1 + exp(- X * w));
y = 2 * (rand(N, 1) < py) - 1;
y_test = 2 * (rand(N_test, 1) < 1 ./ (1 + exp(- X_test * w))) - 1;


%% estimate coefficients, form predictions on train & test sets
% VB logistic regression
[w_VB, V_VB, invV_VB] = vb_logit_fit(X, y);
py_VB = vb_logit_pred(X, w_VB, V_VB, invV_VB);
py_test_VB = vb_logit_pred(X_test, w_VB, V_VB, invV_VB);
% VB logistic regression without hyper-prior
[w_VB1, V_VB1, invV_VB1] = vb_logit_fit_iter(X, y);
py_VB1 = vb_logit_pred(X, w_VB1, V_VB1, invV_VB1);
py_test_VB1 = vb_logit_pred(X_test, w_VB1, V_VB1, invV_VB1);
% Fisher linear discriminant analysis (LDA)
y1 = y == 1;
w_LD = NaN(D, 1);
w_LD(2:end) = (cov(X(y1, 2:end)) + cov(X(~y1, 2:end))) \ ...
              (mean(X(y1, 2:end))' - mean(X(~y1, 2:end))');
w_LD(1) = - 0.5 * (mean(X(y1, 2:end)) + mean(X(~y1, 2:end))) * w_LD(2:end);
y_LD = 2 * (X * w_LD > 0) - 1;
y_test_LD = 2 * (X_test * w_LD > 0) - 1;
% output train and test-set error
fprintf('training set 0-1 loss: LDA = %f, VB = %f, VBiter = %f\n', ...
        mean(y_LD ~= y), mean(2 * (py_VB > 0.5) - 1 ~= y), ...
        mean(2 * (py_VB1 > 0.5) - 1 ~= y));
fprintf('test     set 0-1 loss: LDA = %f, VB = %f, VBiter = %f\n', ...
        mean(y_test_LD ~= y_test), mean(2 * (py_test_VB > 0.5) - 1 ~= y_test), ...
        mean(2 * (py_test_VB1 > 0.5) - 1 ~= y_test));


%% plot true vs. estimated coefficients
f1 = figure;  hold on;
if exist('wlims', 'var'), xlim(wlims); ylim(wlims); end
% error bars
for i = 1:D
    plot(w(i) * [1 1] + 0.02, w_VB(i) + sqrt(V_VB(i,i)) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.8 0.5 0.5]);
    plot(w(i) * [1 1] - 0.02, w_VB1(i) + sqrt(V_VB1(i,i)) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.8 0.5 0.8]);
end
% means
h1 = plot(w + 0.02, w_VB, 'o', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'none');
h2 = plot(w - 0.02, w_VB1, 's', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0.8], 'MarkerEdgeColor', 'none');
h3 = plot(w, w_LD, '+', 'MarkerSize', 3, ...
     'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0 0 0.8], 'LineWidth', 1);
xymin = min([min(xlim) min(ylim)]);  xymax = max([max(xlim) max(ylim)]);
plot([xymin xymax], [xymin xymax], 'k--', 'LineWidth', 0.5);
legend([h1 h2 h3], {'VB', 'VB w/o hyperprior', 'LDA'})
set(gca, 'Box','off', 'PlotBoxAspectRatio', [1 1 1], ...
    'TickDir', 'out', 'TickLength', [1 1]*0.02);
xlabel('w');
ylabel('w_{LD}, w_{VB}');


%% plot separating hyperplanes
f2 = figure; hold on;
% scatterplot of training data, colored by p(y=1)
for i = 1:N
    if y(i) == 1, m = 'o'; else, m = '+'; end
    plot(X(i, 2), X(i, 3), m, 'MarkerSize', 3, 'MarkerFaceColor', 'none', ...
        'MarkerEdgeColor', [0.2 0.4 0.8] + py(i) * [0.6 0 -0.6], ...
        'LineWidth', 0.5);
end
% separating hyperplanes where w1 + w2 x + w3 y = 0
h1 = plot(xlim, -(w(1) + w(2)*xlim) / w(3), 'k-', 'LineWidth', 1);
h2 = plot(xlim, -(w_LD(1) + w_LD(2)*xlim) / w_LD(3), '-', ...
    'LineWidth', 1, 'Color', [0 0 0.8]);
h3 = plot(xlim, -(w_VB(1) + w_VB(2)*xlim) / w_VB(3), '-', ...
    'LineWidth', 1, 'Color', [0.8 0 0]);
h4 = plot(xlim, -(w_VB1(1) + w_VB1(2)*xlim) / w_VB1(3), '-', ...
    'LineWidth', 1, 'Color', [0.8 0 0.8]);
legend([h1 h2 h3 h4], {'true', 'LDA', 'VB', 'VB w/o hyperprior'})
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1],...
    'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('x_2');  ylabel('x_3');