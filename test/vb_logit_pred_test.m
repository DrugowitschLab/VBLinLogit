function vb_logit_pred_test()
%% unit tests for vb_logit_pred(.)

%% settings
wgen = [1 2 3]'; % test weight vector
N_small = 50;    % training set (small)
N_large = 100000;% training set (large)
genX = @(N) bsxfun(@power, linspace(0, 1, N)', 0:(length(wgen)-1));
genY = @(X) 2 * (rand(size(X,1),1) < 1 ./ (1 + exp(-X * wgen))) - 1;


%% test size of returned values
fprintf('Testing vb_logit_pred(.)\n');
fprintf('Size of return values                      ');
X = genX(N_small);
y = genY(X);
[w, V, invV] = vb_logit_fit(X, y);
py = vb_logit_pred(X, w, V, invV);
if ~all(size(py) == [N_small 1])
    fprintf('ERROR: mu size mismatch\n');
else
    fprintf('OK\n');
end


%% test consistency across repeated function calls
fprintf('Consistency across repeated function calls ');
X = genX(N_small);
y = genY(X);
[w, V, invV] = vb_logit_fit(X, y);
py1 = vb_logit_pred(X, w, V, invV);
py2 = vb_logit_pred(X, w, V, invV);
if norm(py1 - py2) > 1e-10
    fprintf('ERROR: mu mismatch\n');
else
    fprintf('OK\n');
end


%% test output predictions
fprintf('Output predictions                         ');
X = genX(N_large);
y = genY(X);
[w, V, invV] = vb_logit_fit(X, y);
ypred = 2 * (vb_logit_pred(X, w, V, invV) > 0.5) - 1;
if mean(abs(ypred - y)) > 0.3
    fprintf('ERROR: ||ypred -  ytrue|| > 1\n');
else
    fprintf('OK\n');
end
