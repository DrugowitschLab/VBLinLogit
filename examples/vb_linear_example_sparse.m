%% sparse linear regression example for vb_linear_*
%
% This example demonstrates the ability of automated relevance
% determination (ARD) to detect and ignore irrelevant input dimensions. The
% script shows this by generating an input -> output mapping with a
% D-dimensional input space, of which only a small subset of D_eff
% dimensions determine the output. It then compares variational Bayesian
% linear regression without and with ARD, and a least-square estimate, and
% shows that the variant with ARD is better able to estimate the regression
% coefficients, and also provides lower-error predictions on the test set. 
%
% Copyright (c) 2014-2019, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% set RNG seed and plot limits to re-produce arXiv figures (in MATLAB)
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave, rand("state", 0); randn("state", 0); else, rng(0); end
wlims = [-5 5];
ylims = [-15 15];


%% settings
D = 1000;        % full input dimensionality
D_eff = 100;     % number of effective input dimensions
N = 500;         % number of training set examples
N_test = 50;     % number of test set examples

% create data
w = [randn(D_eff, 1); zeros(D - D_eff, 1)];
X = rand(N, D) - 0.5;
X_test = rand(N_test, D) - 0.5;
y = X * w + randn(N, 1);
y_test = X_test * w + randn(N_test, 1);


%% preform regression and make predictions
% variational bayes linear regression, train & test-set predictions
[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(X, y);
y_VB = vb_linear_pred(X, w_VB, V_VB, an_VB, bn_VB);
[y_test_VB, lam_VB, nu_VB] = vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB);
% variational bayes linear regression with ARD, train & test-set predictions
[w_VB2, V_VB2, ~, ~, an_VB2, bn_VB2] = vb_linear_fit_ard(X, y);
y_VB2 = vb_linear_pred(X, w_VB2, V_VB2, an_VB2, bn_VB2);
[y_test_VB2, lam_VB2, nu_VB2] = vb_linear_pred(X_test, w_VB2, V_VB2, an_VB2, bn_VB2);
% maximum likelihood, train & test-set predictions
if exist('regress','file') == 2
    [w_ML, wint_ML] = regress(y, X);
else
    w_ML = X \ y;
end
y_ML = X * w_ML;
y_test_ML = X_test * w_ML;
% output train and test set error
fprintf('Training set MSE: ML = %f, VB = %f, VB w/ ARD = %f\n', ...
        mean((y - y_ML).^2), mean((y - y_VB).^2), mean((y - y_VB2).^2));
fprintf('Test     set MSE: ML = %f, VB = %f, VB w/ ARD = %f\n', ...
        mean((y_test - y_test_ML).^2), mean((y_test - y_test_VB).^2), ...
        mean((y_test - y_test_VB2).^2));


%% plot coefficient estimates
f1 = figure;  hold on;
if exist('wlims', 'var'), xlim(wlims); ylim(wlims); end
% error bars
for i = 1:D
    plot(w(i) * [1 1] - 0.02, w_VB(i) + sqrt(V_VB(i,i)) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.8 0.5 0.5]);
    plot(w(i) * [1 1] + 0.02, w_VB2(i) + sqrt(V_VB2(i,i)) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.5 0.8 0.5]);
    if exist('wint_ML','var')
        plot(w(i) * [1 1], wint_ML(i,:), '-', ...
             'LineWidth', 0.25, 'Color', [0.5 0.5 0.8]);
    end
end
% means
h1 = plot(w - 0.02, w_VB, 'o', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'none');
h2 = plot(w + 0.02, w_VB2, 's', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'none');
h3 = plot(w, w_ML, '+', 'MarkerSize', 3, ...
     'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0 0 0.8], 'LineWidth', 1);
xymin = min([min(xlim) min(ylim)]);  xymax = max([max(xlim) max(ylim)]);
plot([xymin xymax], [xymin xymax], 'k--', 'LineWidth', 0.5);
legend([h1 h2 h3], {'VB', 'VB w/ ARD', 'ML'});
set(gca, 'Box','off', 'PlotBoxAspectRatio', [1 1 1],...
    'TickDir', 'out', 'TickLength', [1 1]*0.02);
xlabel('w');
ylabel('w_{ML}, w_{VB}');


%% plot test set predictions
f2 = figure;  hold on;
if exist('ylims', 'var'), xlim(ylims); ylim(ylims); end
y_VB_sd = sqrt((nu_VB/(nu_VB-2))./lam_VB);
y_VB2_sd = sqrt((nu_VB2/(nu_VB2-2))./lam_VB2);
% error bars
for i = 1:N_test
    plot(y_test(i) * [1 1] - 0.02, y_test_VB(i) + y_VB_sd(i) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.8 0.5 0.5]);
    plot(y_test(i) * [1 1] + 0.02, y_test_VB2(i) + y_VB2_sd(i) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.5 0.8 0.5]);
end
% means
h1 = plot(y_test - 0.02, y_test_VB, 'o', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'none');
h2 = plot(y_test + 0.02, y_test_VB2, 's', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'none');
h3 = plot(y_test, y_test_ML, '+', 'MarkerSize', 3, ...
     'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0 0 0.8], 'LineWidth', 1);
xymin = min([min(xlim) min(ylim)]);  xymax = max([max(xlim) max(ylim)]);
plot([xymin xymax], [xymin xymax], 'k--', 'LineWidth', 0.5);
legend([h1 h2 h3], {'VB', 'VB w/ ARD', 'ML'});
set(gca, 'Box','off', 'PlotBoxAspectRatio', [1 1 1], ...
    'TickDir', 'out', 'TickLength', [1 1]*0.02);
xlabel('y');
ylabel('y_{ML}, y_{VB}');
