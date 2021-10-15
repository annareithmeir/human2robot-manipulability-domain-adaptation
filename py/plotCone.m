function plotCone(x, xhat, muall, sigmaall)

% x is 4x400
% xhat is 3x100
% muall is 4x5 modelPD.muMan
% sigmaall is 4x4x5 modelPD.Sigma

nbStates=5;
nbSamples=4;
nbData=100;
out=[2,3,4];

figure('position',[10 10 1800 500],'color',[1 1 1]);
clrmap = lines(nbSamples);
% Plot demonstrations of velocity manipulability ellipsoids in SPD space
subplot(1,3,1); hold on;
title('\fontsize{12}Demonstrations: manipulability ellipsoids');
r = 200;
phi = 0:0.1:2*pi+0.1;
xax = [zeros(size(phi)); r.*ones(size(phi))];
yax = [zeros(size(phi));r.*sin(phi)];
zax = [zeros(size(phi));r/sqrt(2).*cos(phi)]; 
% Cone
h = mesh(xax,yax,zax,'linestyle','none','facecolor',[.95 .95 .95],'facealpha',.5);
direction = cross([1 0 0],[1/sqrt(2),1/sqrt(2),0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(2,:),yax(2,:),zax(2,:),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,63),yax(:,63),zax(:,63),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,40),yax(:,40),zax(:,40),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
% Draw axis
plot3([0,250],[0,0],[0,0],'k','linewidth',.5)
plot3([0,0],[0,250],[0,0],'k','linewidth',.5)
plot3([0,0],[0,0],[0,150],'k','linewidth',.5)
% Add text for axis
text(280,-40,0,'$\mathbf{M}_{11}$','FontSize',20,'Interpreter','latex')
text(15,0,120,'$\mathbf{M}_{12}$','FontSize',20,'Interpreter','latex')
text(5,220,-15,'$\mathbf{M}_{22}$','FontSize',20,'Interpreter','latex')
% Settings
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
axis off
view(70,12);
% Plot demonstrations
for n=1:nbSamples
	for t=1:nbData
        plot3(x(2, t+(n-1)*nbData), x(3, t+(n-1)*nbData), ... 
            x(4, t+(n-1)*nbData)/sqrt(2), '.', 'Markersize', 12, 'color', clrmap(n,:));
    end
end

% Plot demonstrated manipulability and GMM components in SPD space
subplot(1,3,2); hold on;
title('\fontsize{12}Manipulability GMM in SPD space');
clrmap = lines(nbStates);
r = 200;
phi = 0:0.1:2*pi+0.1;
xax = [zeros(size(phi)); r.*ones(size(phi))];
yax = [zeros(size(phi));r.*sin(phi)];
zax = [zeros(size(phi));r/sqrt(2).*cos(phi)]; 
% Cone
h = mesh(xax,yax,zax,'linestyle','none','facecolor',[.95 .95 .95],'facealpha',.5);
direction = cross([1 0 0],[1/sqrt(2),1/sqrt(2),0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(2,:),yax(2,:),zax(2,:),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,63),yax(:,63),zax(:,63),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,40),yax(:,40),zax(:,40),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
% Draw axis
plot3([0,250],[0,0],[0,0],'k','linewidth',.5)
plot3([0,0],[0,250],[0,0],'k','linewidth',.5)
plot3([0,0],[0,0],[0,150],'k','linewidth',.5)
% Add text for axis
text(280,-40,0,'$\mathbf{M}_{11}$','FontSize',20,'Interpreter','latex')
text(15,0,120,'$\mathbf{M}_{12}$','FontSize',20,'Interpreter','latex')
text(5,220,-15,'$\mathbf{M}_{22}$','FontSize',20,'Interpreter','latex')
% Settings
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
axis off
view(70,12);
% Plot samples
for n=1:nbSamples
	for t=1:nbData
        plot3(x(2, t+(n-1)*nbData), x(3, t+(n-1)*nbData), x(4, t+(n-1)*nbData)/sqrt(2), '.', 'Markersize', 10, 'color', [.6 .6 .6]);
    end
end
for i=1:nbStates % Plotting GMM of man. ellipsoids
    %mu = modelPD.MuMan(out,i);
    mu = muall(out,i);
    mu(3) = mu(3)/sqrt(2); % rescale for plots
    %sigma = modelPD.Sigma(out,out,i);
    sigma = sigmaall(out,out,i);
    sigma(3,:) = sigma(3,:)./sqrt(2); % rescale for plots
    sigma(:,3) = sigma(:,3)./sqrt(2); % rescale for plots
    sigma = sigma + 5*eye(3); % for better visualisaton of the last covariance (really thin along one axis)
	plotGMM3D(mu, sigma, clrmap(i,:), .6);
end

subplot(1,3,3); hold on;
title('\fontsize{12}Desired reproduction');
r = 200;
phi = 0:0.1:2*pi+0.1;
xax = [zeros(size(phi)); r.*ones(size(phi))];
yax = [zeros(size(phi));r.*sin(phi)];
zax = [zeros(size(phi));r/sqrt(2).*cos(phi)]; 
% Cone
h = mesh(xax,yax,zax,'linestyle','none','facecolor',[.95 .95 .95],'facealpha',.5);
direction = cross([1 0 0],[1/sqrt(2),1/sqrt(2),0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(2,:),yax(2,:),zax(2,:),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,63),yax(:,63),zax(:,63),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,40),yax(:,40),zax(:,40),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
% Draw axis
plot3([0,250],[0,0],[0,0],'k','linewidth',.5)
plot3([0,0],[0,250],[0,0],'k','linewidth',.5)
plot3([0,0],[0,0],[0,150],'k','linewidth',.5)
% Add text for axis
text(280,-40,0,'$\mathbf{M}_{11}$','FontSize',20,'Interpreter','latex')
text(15,0,120,'$\mathbf{M}_{12}$','FontSize',20,'Interpreter','latex')
text(5,220,-15,'$\mathbf{M}_{22}$','FontSize',20,'Interpreter','latex')
% Settings
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
axis off
view(70,12);
% Plot
plot3(xhat(1, :), xhat(2, :), xhat(3, :)/sqrt(2), '.', 'Markersize', 12, 'color', [0.2 0.8 0.2]);



end