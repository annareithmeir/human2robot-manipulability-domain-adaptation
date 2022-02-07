function A = scaleEllipsoidVolume(A, scale)
    A= (nthroot(scale,3)^2).*A;
end