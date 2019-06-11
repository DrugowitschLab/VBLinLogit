function [w, V, invV, logdetV, an, bn, E_a, L] = vb_linear_fit(X, y, a0, b0, c0, d0)
%% [w, V, invV, logdetV, an, bn, E_a, L] = vb_linear_fit(X, y)
%
% estimates w sucht that y = Xw, using Bayesian regularisation.
%
% The function expects the arguments
% - X: N x D matrix of training input samples, one per row
% - y: N-element column vector of corresponding output samples
% - a0, b0 (optional): scalar prior parameters of the noise precision
% - c0, d0 (optional): scalar hyper-prior shrinkage parameters
% If not given, the prior/hyper-prior parameters default to a0 = 1e-2,
% b0 = 1e-4, c0 = 1e-2, and d0 = 1e-4, resulting in an uninformative prior.
%
% It returns
% - w: posterior weight D-element mean vector
% - V: posterior weight D x D covariance matrix
% - invV, logdetV: inverse of V, and its log-determinant
% - an, bn: scalar posterior parameter of noise precision
% - E_a: scalar mean E(alpha) of shrinkage hyper-posterior
% - L: variational bound, lower-bounding the log-model evidence p(y | X)
%
% The underlying generative model assumes
%
% p(y | x, w, tau) = N(y | w'x, tau^-1),
%
% with x and y being the rows of the given X and y. w and tau are assigned
% the conjugate normal inverse-gamma prior
%
% p(w, tau | alpha) = N(w | 0, (tau alpha)^-1 I) Gam(tau | a0, b0),
%
% with the hyper-prior
%
% p(alpha) = p(alpha | c0, d0).
%
% The returned posterior parameters (computed by variational Bayesian
% inference) determine a posterior of the form
%
% N(w1 | w, tau^-1 V) Gam(tau | an, bn).
%
% Copyright (c) 2013-2019, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% prior parameters
if nargin < 3,  a0 = 1e-2;  end
if nargin < 4,  b0 = 1e-4;  end
if nargin < 5,  c0 = 1e-2;  end
if nargin < 6,  d0 = 1e-4;  end


%% pre-process data
[N, D] = size(X);
X_corr = X' * X;
Xy_corr = X' * y;
an = a0 + N / 2;    gammaln_an = gammaln(an);
cn = c0 + D / 2;    gammaln_cn = gammaln(cn);


%% iterate to find hyperparameters
L_last = -realmax;
max_iter = 500;
E_a = c0 / d0;
for iter = 1:max_iter
    % covariance and weight of linear model
    invV = E_a * eye(D) + X_corr;
    V = inv(invV);
    logdetV = - logdet(invV);
    w = V * Xy_corr;
    % parameters of noise model (an remains constant)
    sse = sum((X * w - y) .^ 2);
    bn = b0 + 0.5 * (sse + E_a * (w' * w));
    E_t = an / bn;

    % hyperparameters of covariance prior (cn remains constant)
    dn = d0 + 0.5 * (E_t * (w' * w) + trace(V));
    E_a = cn / dn;

    % variational bound, ignoring constant terms for now
    L = - 0.5 * (E_t * sse + sum(sum(X .* (X * V)))) + 0.5 * logdetV ...
        - b0 * E_t + gammaln_an - an * log(bn) + an ...
        + gammaln_cn - cn * log(dn);

    % variational bound must grow!
    if L_last > L
        fprintf('Last bound %6.6f, current bound %6.6f\n', L_last, L);
        error('Variational bound should not reduce');
    end
    % stop if change in variation bound is < 0.001%
    if abs(L_last - L) < abs(0.00001 * L)
        break
    end
    L_last = L;    
end
if iter == max_iter
    warning('Bayes:maxIter', ...
        'Bayesian linear regression reached maximum number of iterations.');
end


%% augment variational bound with constant terms
L = L - 0.5 * (N * log(2 * pi) - D) - gammaln(a0) + a0 * log(b0) ...
    - gammaln(c0) + c0 * log(d0);
