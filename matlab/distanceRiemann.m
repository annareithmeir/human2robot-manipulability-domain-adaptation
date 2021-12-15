function d = distanceRiemann(A,B)
    d = sqrt(sum(log(eig(A,B)).^2));
end