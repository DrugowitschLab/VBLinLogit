function vb_linear_pred_test()
%% unit tests for vb_linear_pred(.)

%% settings
wgen = [1 2 3]'; % test weight vector
tau = 2;         % test noise precision
N_small = 50;    % training set (small)
N_large = 10000; % training set (large)
genX = @(N) bsxfun(@power, linspace(0, 1, N)', 0:(length(wgen)-1));
genY = @(X) X * wgen + sqrt(1 / tau) * randn(size(X, 1), 1);


%% test size of returned values
fprintf('Testing vb_linear_pred(.)\n');
fprintf('Size of return values                      ');
X = genX(N_small);
y = genY(X);
[w, V, ~, ~, an, bn] = vb_linear_fit_ard(X, y);
[mu, lambda, nu] = vb_linear_pred(X, w, V, an, bn);
if ~all(size(mu) == [N_small 1])
    fprintf('ERROR: mu size mismatch\n');
elseif ~all(size(lambda) == [N_small 1])
    fprintf('ERROR: lambda size mismatch\n');
elseif ~all(size(nu) == [1 1])
    fprintf('ERROR: nu size mismatch\n');
else
    fprintf('OK\n');
end


%% test consistency across repeated function calls
fprintf('Consistency across repeated function calls ');
X = genX(N_small);
y = genY(X);
[w, V, ~, ~, an, bn] = vb_linear_fit_ard(X, y);
[mu1, lambda1, nu1] = vb_linear_pred(X, w, V, an, bn);
[mu2, lambda2, nu2] = vb_linear_pred(X, w, V, an, bn);
if norm(mu1 - mu2) > 1e-10
    fprintf('ERROR: mu mismatch\n');
elseif norm(lambda1 - lambda2) > 1e-10
    fprintf('ERROR: lambda mismatch\n');
elseif abs(nu1 - nu2) > 1e-10
    fprintf('ERROR: nu mismatch\n');
else
    fprintf('OK\n');
end


%% test output predictions
fprintf('Output predictions                         ');
X = genX(N_large);
y = genY(X);
[w, V, ~, ~, an, bn] = vb_linear_fit_ard(X, y);
mu = vb_linear_pred(X, w, V, an, bn);
if mean((mu - y).^2) > 1
    fprintf('ERROR: ||yest -  ytrue|| > 1\n');
else
    fprintf('OK\n');
end
