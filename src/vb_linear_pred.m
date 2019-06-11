function [mu, lambda, nu] = vb_linear_pred(X, w, V, an, bn)
%% [mu, lambda, nu] = vb_linear_pred(X, w, V, an, bn)
%
% returns the posterior for vb_linear_fit[_ard], given the inputs x being
% the rows of X.
%
% The function expects the arguments
% - X: K x D matrix of K input samples, one per row
% - w: D-element posterior weight mean
% - V: D x D posterior weight covariance matrix
% - an, bn: scalar posterior parameter of noise precision
% w, V, an and bn are the fitted model parameters returned by
% vb_linear_fit[_ard].
%
% It returns
% - mu: K-element predicted output mean vector
% - lambda: K-element predicted output precision vector
% - nu: scalar predicted output degrees of freedom
%
% The arguments are the ones returned by bayes_linear_fit(_ard), specifying
% the parameter posterior
%
% N(w1 | w, tau^-1 V) Gam(tau | an, bn).
%
% The predictive posteriors are of the form
%
% St(y | mu, lambda, nu),
%
% which is a Student's t distribution with mean mu, precision lambda, and
% nu degrees of freedom. All of mu and lambda a vectors, one per input x.
% nu is a scalar as it is the same for all x.
%
% Copyright (c) 2013-2019, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.

mu = X * w;
lambda = (an / bn) ./ (1 + sum(X .* (X * V), 2));
nu = 2 * an;
