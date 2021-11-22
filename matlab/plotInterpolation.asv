%interpolation plot

% pos: nxd array of n points
%scales: 1xn array of n scales
%type: 0,1,2
function interpolation(type)
    positionsHuman = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosHuman.csv");
    scalesHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesHuman.csv");
    positionsRobot = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosRobot.csv");
    scalesRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesRobot.csv");

    [xq,yq,zq] = meshgrid(-0.5:0.1:1);

    figure('position',[10 10 1800 600],'color',[1 1 1]);
    set(gca, 'FontSize', 20)


    % robot plot
    subplot(1,3,1); hold on;
    title('\fontsize{12} Robot');
    axis equal;

     if(type==0)
         Fr = scatteredInterpolant(positionsRobot',scalesRobot', 'linear'); % rowwise positions
     elseif(type==1)
         Fr = scatteredInterpolant(positionsRobot',scalesRobot', 'nearest');
     else
        Fr = scatteredInterpolant(positionsRobot',scalesRobot', 'natural');
     end

    vqr = Fr(xq,yq,zq);
    scatter3(xq(:),yq(:),zq(:),[], vqr(:),'filled');
    colorbar
    shading interp

    % human plot
    subplot(1,3,2); hold on;
    title('\fontsize{12} Human');
    axis equal;

     if(type==0)
         Fh = scatteredInterpolant(positionsHuman',scalesHuman', 'linear'); % rowwise positions
     elseif(type==1)
         Fh = scatteredInterpolant(positionsHuman',scalesHuman', 'nearest');
     else
        Fh = scatteredInterpolant(positionsHuman',scalesHuman', 'natural');
     end

    vqh = Fh(xq,yq,zq);
    scatter3(xq(:),yq(:),zq(:),[], vqh(:),'filled');
    colorbar
    shading interp


    % diff plot
    subplot(1,3,3); hold on;
    title('\fontsize{12} Diff');
    axis equal;

    vq = abs(vqr-vqh);
    scatter3(xq(:),yq(:),zq(:),[], vq(:),'filled');
    colorbar
    shading interp
    
    saveas(gcf,'/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/plots/interpolation.png')
    
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
