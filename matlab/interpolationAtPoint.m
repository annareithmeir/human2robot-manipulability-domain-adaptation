%interpolation of 3d-1d irregular positions and scales

% pos: nxd array of n points
%scales: 1xn array of n scales
%type: 0,1,2
function vq=interpolationAtPoint(positions, scales, x,y,z, type)
    %positions = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPos.csv");
    %scales=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScales.csv");


     if(type==0)
         F = scatteredInterpolant(positions',scales', 'linear'); % rowwise positions
     elseif(type==1)
         F = scatteredInterpolant(positions',scales', 'nearest');
     else
        F = scatteredInterpolant(positions',scales', 'natural');
     end

    vq= F(x,y,z);


    % plotting
    % [xq,yq,zq] = meshgrid(-0.5:0.025:1)
    % vq = F(xq,yq,zq)
    % 
    % [xq,yq,zq] = meshgrid(-0.5:0.025:1);
    % xslice = [-.5,0,.5]; 
    % yslice = [0,0.5]; 
    % zslice = [-.5,.5];
    % slice(xq,yq,zq,vq,xslice,yslice,zslice)
end
