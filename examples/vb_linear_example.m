%% simple script demonstrating the use of vb_linear_fit.m and
% vb_linear_fit_ard.m
%
% This script tests linear regression on two generated dataset. Both
% feature linear input -> output mappings, but only noisy outputs are
% observed. They differ in the dimensionality of the inputs. The first
% features a d-dimensional input, whereas the second has an additional
% d_extra dimensions that are uninformative about the output values.
%
% The script compares multiple linear regression approaches on these
% datasets:
% - Least-squares / maximum likelihood regression
% - Variational Bayesian linear regression
% - Variational Bayesian linear regression with Automated Relevance
%   Determination (ARD)
% The maximum likelihood approach is expected to overfit, in particular in
% the presence of uninformative dimensions. The method with ARD is expected
% to be most robust against addition of such uninformative dimensions.
%
% Copyright (c) 2013, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% set RNG seed
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave, rand("state", 1); randn("state", 1); else, rng(4); end


%% dimensionality, number of data points, noise
d = 4;           % base dimensionality of X -> y mapping
d_extra = 10;    % additional, uninformative dimensions
N = 50;          % number of data points in training set
N_cv = 50;       % number of data points in test set
tau = 1;         % inverse variance of additive noise


%% random weight vector & predictions
w = randn(d, 1);
% inputs for train/test set
x = rand(N, 1);
X = bsxfun(@power, x, 0:(d-1));
X_ext = bsxfun(@power, x, 0:(d + d_extra - 1));
X_cv = bsxfun(@power, rand(N_cv, 1), 0:(d + d_extra - 1));
% corresponding noise-free and noisy outputs
y_no_noise = X * w;
y = y_no_noise + sqrt(1 / tau) * randn(N, 1);
y_cv = X_cv(:, 1:d) * w;
% inputs used to plot predictions
x_pred = linspace(0, 1, 100)';
X_pred = bsxfun(@power, x_pred, 0:(d-1));
X_pred_ext = bsxfun(@power, x_pred, 0:(d+d_extra-1));


%% Estimate weights by least-squares / maximum likelihood
w_ML = X \ y;
w_ML_ext = X_ext \ y;
% predictions for training and test set on extended input space
y_ML = X_ext * w_ML_ext;
y_ML_cv = X_cv * w_ML_ext;


%% weights and predictions by variational bayes (only on extended space)
[w_vb, V_vb, ~, ~, an_vb, bn_vb] = vb_linear_fit(X_ext, y);
[y_vb, lam_vb, nu_vb] = vb_linear_pred(X_pred_ext, w_vb, V_vb, an_vb, bn_vb);
% standard deviation of Student's t posterior with parameters lam_vb and nu_vb
y_vb_sd = sqrt(nu_vb ./ (lam_vb .* (nu_vb - 2)));

% the same with ARD
[w_vb_ard, V_vb, invV_vb, logdetV_vb, an_vb, bn_vb] = vb_linear_fit_ard(X_ext, y);
[y_vb_ard, lam_vb, nu_vb] = vb_linear_pred(X_pred_ext, w_vb_ard, V_vb, an_vb, bn_vb);
% standard deviation of Student's t posterior with parameters lam_vb and nu_vb
y_vb_ard_sd = sqrt(nu_vb ./ (lam_vb .* (nu_vb - 2)));


%% plot fits
f1 = figure; set(0,'defaultlinelinewidth', 1.5);  hold on;
plot(x_pred, X_pred * w, 'k-');
plot(x, y, 'k+');
plot(x_pred, X_pred * w_ML, 'g-');
plot(x_pred, X_pred_ext * w_ML_ext, 'g--');
plot(x_pred, y_vb, 'r-');
plot(x_pred, y_vb_ard, 'b-');
plot(x_pred, y_vb + y_vb_sd, 'r-.');
plot(x_pred, y_vb - y_vb_sd, 'r-.');
plot(x_pred, y_vb_ard + y_vb_ard_sd, 'b-.');
plot(x_pred, y_vb_ard - y_vb_ard_sd, 'b-.');
legend('true', 'data', 'ML', 'ML ext', 'vb ext', 'vb ext ard');
set(gca,'TickDir','out');
xlabel('x');
ylabel('y');

% print training set and cross-validated MSE
fprintf('MSEs:       training set     test set\n');
fprintf('ML          %7.5f          %7.5f\n', ...
        mean((y - X_ext * w_ML_ext).^2), mean((y_cv - X_cv * w_ML_ext).^2));
fprintf('VB          %7.5f          %7.5f\n', ...
        mean((y - X_ext * w_vb).^2), mean((y_cv - X_cv * w_vb).^2));
fprintf('VB (ARD)    %7.5f          %7.5f\n', ...
        mean((y - X_ext * w_vb_ard).^2), mean((y_cv - X_cv * w_vb_ard).^2));
