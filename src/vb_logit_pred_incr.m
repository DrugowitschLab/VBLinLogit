function out = vb_logit_pred_incr(X, w, V, invV)
%% out = vb_logit_pred_incr(X, w, V, invV)
%
% returns a vector containing p(y=1 | x, X, Y) for x = each row in the
% given X, for a fitted Bayesian logit model.
%
% The function expects the arguments
% - X: K x D matrix of K input samples, one per row
% - w: D-element posterior weight mean
% - V: D x D posterior weight covariance matrix
% - invV: inverse of V
% w, V and invV are the fitted model parameters returned by
% vb_logit_fit[_*].
%
% It returns
% - out: K-element vector, with p(y=1 | x, X, Y) as each element.
%
% The function assumes model parameters corresponding to the data
% likelihood
%
% p(y = 1 | x, w1) = 1 / (1 + exp(- w1' * x)),
%
% with w, V, invV specifying the posterior parameters N(w1 | w, V). In
% constrast to vb_logit_pred, which computes the predictions for all rows of
% X simultaneously, this function interates over the rows of X.
%
% Copyright (c) 2013, 2014, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.

max_iter = 500;
N = size(X, 1);
out = zeros(N, 1);


%% iterate over x, finding xi for each x separately
for n = 1:N;
    xn = X(n,:)';

    % precompute values
    Vx = V * xn;
    VxVx = Vx * Vx';
    c = xn' * Vx;
    xx = xn * xn';
    t_w = invV * w + 0.5 * xn;

    % start iteration at xi = 0, lam_xi = 1/8
    V_xi = V - VxVx / (4 + c);
    invV_xi = invV + xx / 4;
    logdetV_xi = -log(1 + c / 4);
    w_xi = V_xi * t_w;
    L_last = 0.5 * (logdetV_xi + w_xi' * invV_xi * w_xi) - log(2);

    % iterate to find xi that maximises variational bound
    for i = 1:max_iter
        % update xi by EM algorithm
        xi = sqrt(xn' * (V_xi + w_xi * w_xi') * xn);
        lam_xi = lam(xi);

        % Sherman-Morrison formula and Matrix determinant lemma
        V_xi = V - (2 * lam_xi / (1 + 2 * lam_xi * c)) * VxVx;
        invV_xi = invV + 2 * lam_xi * xx;
        logdetV_xi = -log(1 + 2 * lam_xi * c);
        w_xi = V_xi * t_w;

        % variational bound, omitting constant terms
        L = 0.5 * (logdetV_xi + w_xi' * invV_xi * w_xi - xi) ...
            - log(1 + exp(- xi)) + lam_xi * xi^2;

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
   
    % p(y=1 | x, X, Y), using again Matrix determinant lemma
    out(n) = 1 / (1 + exp(-xi)) / sqrt(1 + 2 * lam_xi * c) ...
             * exp(- 0.5 * xi + lam_xi * xi ^ 2 ...
                   - 0.5 * w' * invV * w + 0.5 * w_xi' * invV_xi * w_xi);
end


function out = lam(xi)
% returns 1 / (4 * xi) * tanh(xi / 2)
if xi == 0
    out = 1 / 8;
else
    out = tanh(xi / 2) / (4 * xi);
end
