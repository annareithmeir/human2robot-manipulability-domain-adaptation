%interpolation plot

% pos: nxd array of n points
%scales: 1xn array of n scales
%type: 0,1,2
function plotInterpolation(type)
    positionsHuman = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_positions.csv");
    scalesHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_scales_normalized.csv");
    manipsHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_manipulabilities_normalized.csv");
    positionsRobot = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_positions.csv");
    scalesRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_scales_normalized.csv");
    manipsRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_manipulabilities_normalized.csv");

    [xq,yq,zq] = meshgrid(-0.5:0.1:1);

    figure('position',[10 10 1800 600],'color',[1 1 1]);
    set(gca, 'FontSize', 20)

    % positions plot
    subplot(1,4,1); hold on;
    title('\fontsize{12} Random config positions (red: human, blue:robot)');
    axis equal;
    
    scatter3(positionsHuman(:,1),positionsHuman(:,2),positionsHuman(:,3), 'red','x');
    scatter3(positionsRobot(:,1),positionsRobot(:,2),positionsRobot(:,3),  'blue','x');
    alpha(.2)
      
    
    
    % robot plot
    subplot(1,4,2); hold on;
    title('\fontsize{12} Robot');
    axis equal;

     if(type==0)
         Fr = scatteredInterpolant(positionsRobot,scalesRobot, 'linear'); % rowwise positions
     elseif(type==1)
         Fr = scatteredInterpolant(positionsRobot,scalesRobot, 'nearest');
     else
        Fr = scatteredInterpolant(positionsRobot,scalesRobot, 'natural');
     end

    vqr = Fr(xq,yq,zq);
    scatter3(xq(:),yq(:),zq(:),[], vqr(:),'filled','SizeData',8);
    colorbar
    shading interp
    alpha(.5)

    % human plot
    subplot(1,4,3); hold on;
    title('\fontsize{12} Human');
    axis equal;

     if(type==0)
         Fh = scatteredInterpolant(positionsHuman,scalesHuman, 'linear'); % rowwise positions
     elseif(type==1)
         Fh = scatteredInterpolant(positionsHuman,scalesHuman, 'nearest');
     else
        Fh = scatteredInterpolant(positionsHuman,scalesHuman, 'natural');
     end

    vqh = Fh(xq,yq,zq);
    scatter3(xq(:),yq(:),zq(:),[], vqh(:),'filled','SizeData',8);
    colorbar
    shading interp
    alpha(.5)


    % diff plot
    subplot(1,4,4); hold on;
    title('\fontsize{12} Diff');
    axis equal;

    vq = abs(vqr-vqh);
    scatter3(xq(:),yq(:),zq(:),[], vq(:),'filled','SizeData',8);
    colorbar
    shading interp
    alpha(.5)
    
    saveas(gcf,'/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/plots/interpolation.png')
    
end
