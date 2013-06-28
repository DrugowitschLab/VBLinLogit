function res = logdet(A)
% LOGDET(A) computes the log(det(A)) of a positive definite A.
% This function is more accurate than log(det(A))
res = 2 * sum(log(diag(chol(A))), 1);