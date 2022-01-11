function [Mn, scale] = normalize_manipulability(M)
    scale=prod(sqrt(eig(M)))*(4.0/3.0)*pi;
    Mn = scaleEllipsoidVolume(M, 1/scale);
end
