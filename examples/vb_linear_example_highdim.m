%% high-dimensional linear regression example for vb_linear_*
%
% Copyright (c) 2014, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% set RNG seed and plot limits to re-produce arXiv figures (in MATLAB)
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave, rand("state", 0); else, rng(0); end
wlims = [-5 5];
ylims = [-11 11];


%% settings
D = 100;
N = 150;
N_test = 50;
% create data
w = randn(D, 1);
X = rand(N, D) - 0.5;
X_test = rand(N_test, D) - 0.5;
y = X * w + randn(N, 1);
y_test = X_test * w + randn(N_test, 1);


%% preform regression and make predictions
% variational bayes linear regression
[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(X, y);
y_VB = vb_linear_pred(X, w_VB, V_VB, an_VB, bn_VB);
[y_test_VB, lam_VB, nu_VB] = vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB);
% maximum likelihood
if exist('regress','file') == 2
    [w_ML, wint_ML] = regress(y, X);
else
    w_ML = X \ y;
end
y_ML = X * w_ML;
y_test_ML = X_test * w_ML;
% train and test set error
fprintf('Training set MSE: ML = %f, VB = %f\n', ...
        mean((y - y_ML).^2), mean((y - y_VB).^2));
fprintf('Test     set MSE: ML = %f, VB = %f\n', ...
        mean((y_test - y_test_ML).^2), mean((y_test - y_test_VB).^2));


%% plot coefficient estimates
f1 = figure;  hold on;
if exist('wlims', 'var'), xlim(wlims); ylim(wlims); end
% error bars
for i = 1:D
    plot(w(i) * [1 1] - 0.01, w_VB(i) + sqrt(V_VB(i,i)) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.8 0.5 0.5]);
    if exist('wint_ML','var')
        plot(w(i) * [1 1] + 0.01, wint_ML(i,:), '-', ...
             'LineWidth', 0.25, 'Color', [0.5 0.5 0.8]);
    end
end
% means
plot(w - 0.01, w_VB, 'o', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'none');
plot(w + 0.01, w_ML, '+', 'MarkerSize', 3, ...
     'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0 0 0.8], 'LineWidth', 1);
xymin = min([min(xlim) min(ylim)]);  xymax = max([max(xlim) max(ylim)]);
plot([xymin xymax], [xymin xymax], 'k--', 'LineWidth', 0.5);
set(gca, 'Box','off', 'PlotBoxAspectRatio', [1 1 1],...
    'TickDir', 'out', 'TickLength', [1 1]*0.02);
xlabel('w');  ylabel('w_{ML}, w_{VB}');


%% plot test set predictions
f2 = figure;  hold on;
if exist('ylims', 'var'), xlim(ylims); ylim(ylims); end
y_VB_sd = sqrt(nu_VB ./ (lam_VB .* (nu_VB - 2)));
for i = 1:N_test
    plot(y_test(i) * [1 1], y_test_VB(i) + y_VB_sd(i) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.8 0.5 0.5]);
end
plot(y_test, y_test_VB, 'o', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'none');
plot(y_test, y_test_ML, '+', 'MarkerSize', 3, ...
     'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0 0 0.8], 'LineWidth', 1);
xymin = min([min(xlim) min(ylim)]);  xymax = max([max(xlim) max(ylim)]);
plot([xymin xymax], [xymin xymax], 'k--', 'LineWidth', 0.5);
set(gca, 'Box','off', 'PlotBoxAspectRatio', [1 1 1],...
    'TickDir', 'out', 'TickLength', [1 1]*0.02);
xlabel('y');  ylabel('y_{ML}, y_{VB}');
