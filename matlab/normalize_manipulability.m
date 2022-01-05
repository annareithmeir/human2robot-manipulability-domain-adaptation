function [Mn, scale] = normalize_manipulability(M)
    M
    eig(M)

    scale=prod(sqrt(eig(M)))*(4.0/3.0)*pi;
    scale
    Mn = scaleEllipsoidVolume(M, 1/scale);
end
