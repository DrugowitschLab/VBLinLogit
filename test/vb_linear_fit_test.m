function vb_linear_fit_test()
%% unit tests for vb_linear_fit(.)

%% settings
wgen = [1 2 3]'; % test weight vector
tau = 2;         % test noise precision
N_small = 50;    % training set (small)
N_large = 10000; % training set (large)
a0 = 1e-2;
b0 = 1e-4;
c0 = 1e-2;
d0 = 1e-4;
genX = @(N) bsxfun(@power, linspace(0, 1, N)', 0:(length(wgen)-1));
genY = @(X) X * wgen + sqrt(1 / tau) * randn(size(X, 1), 1);


%% test size of returned values
fprintf('Testing vb_linear_fit(.)\n');
fprintf('Size of return values                      ');
X = genX(N_small);
y = genY(X);
[w, V, invV, logdetV, an, bn, E_a, L] = vb_linear_fit(X, y);
if ~all(size(w) == size(wgen))
    fprintf('ERROR: w size mismatch\n');
elseif ~all(size(V) == [1 1]*length(wgen))
    fprintf('ERROR: V size mismatch\n');
elseif ~all(size(invV) == [1 1]*length(wgen))
    fprintf('ERROR: invV size mismatch\n');
elseif ~all(size(logdetV) == [1 1])
    fprintf('ERROR: logdetV size mismatch\n');
elseif ~all(size(an) == [1 1])
    fprintf('ERROR: an size mismatch\n');
elseif ~all(size(bn) == [1 1])
    fprintf('ERROR: bn size mismatch\n');
elseif ~all(size(E_a) == [1 1])
    fprintf('ERROR: E_a size mismatch\n');
elseif ~all(size(L) == [1 1])
    fprintf('ERROR: L size mismatch\n');
else
    fprintf('OK\n');
end


%% test consistency across repeated function calls
fprintf('Consistency across repeated function calls ');
X = genX(N_small);
y = genY(X);
[w1, V1, invV1, logdetV1, an1, bn1, E_a1, L1] = vb_linear_fit(X, y);
[w2, V2, invV2, logdetV2, an2, bn2, E_a2, L2] = vb_linear_fit(X, y);
if norm(w1 - w2) > 1e-10
    fprintf('ERROR: w mismatch\n');
elseif norm(V1 - V2) > 1e-10
    fprintf('ERROR: V mismatch\n');
elseif norm(invV1 - invV2) > 1e-10
    fprintf('ERROR: invV mismatch\n');
elseif abs(logdetV1 - logdetV2) > 1e-10
    fprintf('ERROR: logdetV mismatch\n');
elseif abs(an1 - an2) > 1e-10
    fprintf('ERROR: an mismatch\n');
elseif abs(bn1 - bn2) > 1e-10
    fprintf('ERROR: bn mismatch\n');
elseif abs(E_a1 - E_a2) > 1e-10
    fprintf('ERROR: E_a mismatch\n');
elseif abs(L1 - L2) > 1e-10
    fprintf('ERROR: L mismatch\n');
else
    fprintf('OK\n');
end


%% test optional arguments
fprintf('Use of optional arguments                  ');
X = genX(N_small);
y = genY(X);
[w1, V1] = vb_linear_fit(X, y);
[w2, V2] = vb_linear_fit(X, y, a0);
[w3, V3] = vb_linear_fit(X, y, a0, b0);
[w4, V4] = vb_linear_fit(X, y, a0, b0, c0);
[w5, V5] = vb_linear_fit(X, y, a0, b0, c0, d0);
if norm(w1 - w2) > 1e-10 || norm(V1 - V2) > 1e-10
    fprintf('ERROR: mismatch with a0 argument\n');
elseif norm(w1 - w3) > 1e-10 || norm(V1 - V3) > 1e-10
    fprintf('ERROR: mismatch with b0 argument\n');
elseif norm(w1 - w4) > 1e-10 || norm(V1 - V4) > 1e-10
    fprintf('ERROR: mismatch with b0 argument\n');
elseif norm(w1 - w5) > 1e-10 || norm(V1 - V5) > 1e-10
    fprintf('ERROR: mismatch with b0 argument\n');
else
    fprintf('OK\n');
end


%% test weight estimates
fprintf('Weight estimates                           ');
X = genX(N_large);
y = genY(X);
w = vb_linear_fit(X, y);
if norm(w - wgen) > 0.5
    fprintf('ERROR: ||w -  wtrue|| > 0.5\n');
else
    fprintf('OK\n');
end
