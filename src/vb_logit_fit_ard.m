function [w, V, invV, logdetV, E_a, L] = vb_logit_fit_ard(X, y, a0, b0)
%% [w, V, invV, logdetV, E_a, L] = vb_logit_fit_ard(X, y)
%
% returns parpameters of a fitted logit model
%
% p(y = 1 | x, w) = 1 / (1 + exp(- w' * x)),
%
% with and automatic relevance determination on w.
%
% The function expects the arguments
% - X: N x D matrix of training input samples, one per row
% - y: N-element column vector of corresponding output {-1, 1} samples
% - a0, b0 (optional): scalar shrinkage prior parameters
% If not given, the prior/hyper-prior parameters default to a0 = 1e-2,
% b0 = 1e-4, resulting in an weak shrinkage prior.
%
% It returns
% - w: posterior weight D-element mean vector
% - V: posterior weight D x D covariance matrix
% - invV, logdetV: inverse of V, and its log-determinant
% - E_a: mean vector E(a) of shrinkage posteriors
% - L: variational bound, lower-bounding the log-model evidence p(y | X)
%
% The underlying generative model assumes a weight vector prior
%
% p(w_i | a_i) = N(w_i | 0, a_i^-1)
%
% and hyperpriors
%
% p(a_i) = Gam(a_i | a0, b0).
%
% The function returns the parameters of the posterior
%
% p(w1 | X, y) = N(w1 | w, V).
%
% Copyright (c) 2013-2019, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% hyperprior parameters
if nargin < 3,  a0 = 1e-2;  end
if nargin < 4,  b0 = 1e-4;  end


%% pre-compute some constants
[N, D] = size(X);
max_iter = 500;
an = a0 + 0.5;    D_gammaln_an_an = D * (gammaln(an) + an);
t_w = 0.5 * sum(bsxfun(@times, X, y), 1)';


%% start first iteration kind of here, with xi = 0 -> lam_xi = 1/8
lam_xi = ones(N, 1) / 8;
E_a = ones(D, 1) * a0 / b0;
invV = diag(E_a) + 2 * X' * bsxfun(@times, X, lam_xi);
V = inv(invV);
w = V * t_w;
bn = b0 + 0.5 * (w .^ 2 + diag(V));
L_last = - N * log(2) ...
         + 0.5 * (w' * invV * w - logdet(invV)) ...
         - sum((b0 * an) ./ bn) - sum(an * log(bn)) + D_gammaln_an_an;


%% update xi, bn, (V, w) iteratively
for i = 1:max_iter
    % update xi by EM-algorithm
    xi = sqrt(sum(X .* (X * (V + w * w')), 2));
    lam_xi = lam(xi);

    % update posterior parameters of a based on xi
    bn = b0 + 0.5 * (w .^ 2 + diag(V));
    E_a = an ./ bn;

    % recompute posterior parameters of w
    invV = diag(E_a) + 2 * X' * bsxfun(@times, X, lam_xi);
    V = inv(invV);
    logdetV = - logdet(invV);
    w = V * t_w;

    % variational bound, ingnoring constant terms for now
    L = - sum(log(1 + exp(- xi))) + sum(lam_xi .* xi .^ 2) ...
        + 0.5 * (w' * invV * w + logdetV - sum(xi)) ...
        - sum(b0 * E_a) - sum(an * log(bn)) + D_gammaln_an_an;

    % either stop if variational bound grows or change is < 0.001%
    % HACK ALARM: theoretically, the bound should never grow, and it doing
    % so points to numerical instabilities. As it seems, these start to
    % occur close to the optimal bound, which already points to a good
    % approximation.    
    if (L_last > L) || (abs(L_last - L) < abs(0.00001 * L))
        break
    end
    L_last = L;  
end;
if i == max_iter
    warning('Bayes:maxIter', ...
        'Bayesian logistic regression reached maximum number of iterations.');
end

%% add constant terms to variational bound
L = L - D * (gammaln(a0) - a0 * log(b0));


function out = lam(xi)
% returns 1 / (4 * xi) * tanh(xi / 2)
divby0_w = warning('query', 'MATLAB:divideByZero');
warning('off', 'MATLAB:divideByZero');
out = tanh(xi ./ 2) ./ (4 .* xi);
warning(divby0_w.state, 'MATLAB:divideByZero');
% fix values where xi = 0
out(isnan(out)) = 1/8;
