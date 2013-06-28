function [w, V, invV, logdetV, E_a, L] = bayes_logit_fit_ard(X, y)
%% [w, V, invV, logdetV, E_a, L] = bayes_logit_fit_ard(X, y)
%
% returns parpameters of a fitted logit model
%
% p(y = 1 | x, w) = 1 / (1 + exp(- w' * x)),
%
% with weight vector prior
%
% p(w_i | a_i) = N(w_i | 0, a_i^-1)
%
% and hyperprior
%
% p(a_i) = Gam(a_i | a0, b0).
%
% The parameters a0 and b0 are set such that the prior is uninformative.
%
% The arguments are:
% X - input matrix, inputs x as row vectors
% y - output vector, containing either 1 or -1
%
% The function returns the posterior p(w1 | X, y) = N(w1 | w, V), and 
% additionally the inverse of V and ln|V| (just in case). The returned vector
% E_a is the expectations of the posterior a_i's. L is the final variational
% bound, which is a lower bound on the log-model evidence.

%% hyperprior parameters
a0 = 1e-2;
b0 = 1e-4;

% equations from Bishop (2006) PRML Book + errata (!) + new stuff


%% pre-compute some constants
[N, D] = size(X);
max_iter = 100;
an = a0 + 0.5;    D_gammaln_an_an = D * (gammaln(an) + an);
% t_W = 0.5 * sum(X .* repmat(y, 1, D), 1)'
t_w = 0.5 * sum(bsxfun(@times, X, y), 1)';


%% start first iteration kind of here, with xi = 0 -> lam_xi = 1/8
lam_xi = ones(N, 1) / 8;
E_a = ones(D, 1) * a0 / b0;
% invV = diag(E_a) + 2 * X' * (X .* repmat(lam_xi, 1, D))
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
    % invV = diag(E_a) + 2 * X' * (X .* repmat(lam_xi, 1, D))
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