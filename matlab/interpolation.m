%interpolation plot

% pos: nxd array of n points
%scales: 1xn array of n scales
%type: 0,1,2
function [xq,yq,zq,vq]=interpolation(positions, scales, type)
    %positions = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPos.csv");
    %scales=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScales.csv");


     if(type==0)
         F = scatteredInterpolant(positions',scales', 'linear'); % rowwise positions
     elseif(type==1)
         F = scatteredInterpolant(positions',scales', 'nearest');
     else
        F = scatteredInterpolant(positions',scales', 'natural');
     end


    % plotting
    [xq,yq,zq] = meshgrid(-0.5:0.1:1);
    vq = F(xq,yq,zq);
    scatter3(xq(:),yq(:),zq(:),[], vq(:),'filled');
    colorbar
    shading interp
    
%     Slice map
%     [xq,yq,zq] = meshgrid(-0.5:0.05:1);
%     xslice = [0,.5]; 
%     yslice = [0,0.5]; 
%     zslice = [0,.5];
%     slice(xq,yq,zq,vq,xslice,yslice,zslice);

%     Only random samples without interpolation
%     x=positions(1,:);
%     y=positions(2,:);
%     z=positions(3,:);
%     scatter3(x,y,z,[],scales)
end