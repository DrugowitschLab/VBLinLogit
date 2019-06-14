function vb_logit_fit_iter_test()
%% unit tests for vb_logit_fit_iter(.)

%% settings
wgen = [1 2 3]'; % test weight vector
N_small = 50;    % training set (small)
N_large = 100000; % training set (large)
genX = @(N) bsxfun(@power, linspace(0, 1, N)', 0:(length(wgen)-1));
genY = @(X) 2 * (rand(size(X,1),1) < 1 ./ (1 + exp(-X * wgen))) - 1;


%% test size of returned values
fprintf('Testing vb_logit_fit_iter(.)\n');
fprintf('Size of return values                      ');
X = genX(N_small);
y = genY(X);
[w, V, invV, logdetV] = vb_logit_fit_iter(X, y);
if ~all(size(w) == size(wgen))
    fprintf('ERROR: w size mismatch\n');
elseif ~all(size(V) == [1 1]*length(wgen))
    fprintf('ERROR: V size mismatch\n');
elseif ~all(size(invV) == [1 1]*length(wgen))
    fprintf('ERROR: invV size mismatch\n');
elseif ~all(size(logdetV) == [1 1])
    fprintf('ERROR: logdetV size mismatch\n');
else
    fprintf('OK\n');
end


%% test consistency across repeated function calls
fprintf('Consistency across repeated function calls ');
X = genX(N_small);
y = genY(X);
[w1, V1, invV1, logdetV1] = vb_logit_fit_iter(X, y);
[w2, V2, invV2, logdetV2] = vb_logit_fit_iter(X, y);
if norm(w1 - w2) > 1e-10
    fprintf('ERROR: w mismatch\n');
elseif norm(V1 - V2) > 1e-10
    fprintf('ERROR: V mismatch\n');
elseif norm(invV1 - invV2) > 1e-10
    fprintf('ERROR: invV mismatch\n');
elseif abs(logdetV1 - logdetV2) > 1e-10
    fprintf('ERROR: logdetV mismatch\n');
else
    fprintf('OK\n');
end


%% test weight estimates
fprintf('Weight estimates                           ');
X = genX(N_large);
y = genY(X);
w = vb_logit_fit_iter(X, y);
if norm(w - wgen) > 1
    fprintf('ERROR: ||w -  wtrue|| > 1\n');
else
    fprintf('OK\n');
end
