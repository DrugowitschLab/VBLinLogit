function out = bayes_logit_post_incr(X, w, V, invV)
% BAYES_LOGIT_POST_INCR returns a vector containing p(y=1 | x, X, Y) for x =
% each row in the given X, for a bayesian logit model with p(y = 1 | x, w)
% = 1 / (1 + exp(- w' * x)), and w, V, invV, logdetV are the posterior
% parameters N(w, V). In constrast to BAYES_LOGIT_POST, this function
% interates over the rows of X.

max_iter = 100;
N = size(X, 1);
out = zeros(N, 1);

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

    for i = 1:max_iter
        % update xi by EM algorithm
        xi = sqrt(xn' * (V_xi + w_xi * w_xi') * xn);
        lam_xi = lam(xi);
        % Sherman-Morrison formula
        V_xi = V - (2 * lam_xi / (1 + 2 * lam_xi * c)) * VxVx;
        invV_xi = invV + 2 * lam_xi * xx;
        logdetV_xi = -log(1 + 2 * lam_xi * c);
        w_xi = V_xi * t_w;
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
   
    % p(y=1 | x, X, Y)
    out(n) = 1 / (1 + exp(-xi)) / sqrt(1 + 2 * lam_xi * c) ...
             * exp(- xi / 2 + lam_xi * xi ^ 2 ...
                   - w' * invV * w / 2 + w_xi' * invV_xi * w_xi / 2);
end


function out = lam(xi)
% returns 1 / (4 * xi) * tanh(xi / 2)
if xi == 0
    out = 1 / 8;
else
    out = tanh(xi / 2) / (4 * xi);
end