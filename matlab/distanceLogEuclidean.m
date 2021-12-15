function d = distanceLogEuclidean(A,B)
    d = norm(logm(A) - logm(B), 'fro');
end