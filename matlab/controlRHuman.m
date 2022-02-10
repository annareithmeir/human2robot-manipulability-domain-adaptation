% % Control rHuman model towards q position
% 
% data_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human/reach_up"
% 
% rhuman = rHuManModel('shoulderHeight',1.35,'verbose',true);
% theta = [90; 30; 0; 40;   0; 0; 0]*pi/180;
% % rhuman.plot(theta);
% % view(-70,22);
% % axis([-0.6 0.6 -0.3 0.6 0.0 1.8]);
% 
% position = rhuman.getPos(theta)
% orientation = rhuman.getOrientation(theta)
% 
% e=0.025;
% e_cnt = 9999;
% q_desired = [90; 30; 0;     0;     0; 0; 0]*pi/180; 
% x_desired = rhuman.getFKM(q_desired);
% q_cnt=theta;
% 
% transDesired = [0; 0.65; rhuman.kine.base.translation.q(4)];
% 
% FLAG_ACT = false;
% 
% while (norm(e_cnt) > e)
%     rhuman.plot(q_cnt);
%     view(-70,22);
%     axis([-0.6 0.6 -0.3 0.6 0.0 1.8]);
%     
%     x = rhuman.getFKM(q_cnt);
%     %e_cnt = vec8(x-x_desired);
%     
% %     e_cnt = (x.translation.q(2:4) - x_desired.translation.q(2:4));
%     e_cnt = (x.translation.q(2:4) - transDesired);
%     
%     J = rhuman.getJacobGeom(q_cnt);
%     J_t = J(4:6,:); 
% 
%     
%     [U,S,V] = svd(J_t); 
%     svd(J_t)' 
%     x.translation.q(2:4)' 
%     
% %     pseudoinverse =  J_t'*pinv( (J_t*J_t') + 1*eye(3));
%     pseudoinverse =  J_t'*pinv( (J_t*J_t'));
%     
%     u = -0.01 *pinv(J_t) * e_cnt;
%     if (S(3,3) < 0.06)
%         FLAG_ACT = true;
%     end
%     if FLAG_ACT
%         u = -0.01 * (eye(7) - V(3:end,:)'*V(3:end,:))  *pinv(J_t) * e_cnt;
%     end
%     q_cnt = q_cnt + u;
% %     norm(e_cnt)
% end




% figure; 
% rhuman = rHuManModel('shoulderHeight',1.35,'verbose',true);
% theta=zeros(7,1); 
% theta(2)=150*pi/180; %100/90/80
% n_points=20;
% 
% manipulabilities=zeros(n_points, 9);
% manipulabilities_normalized=zeros(n_points, 9); % all volume = 1
% scales=zeros(1,n_points);
% format long
% 
% rhuman.plot(theta); 
% view(-79,22); 
% axis([-0.6 0.6 -0.3 0.6 0.0 1.8]); 
% 
% % Rotation the first joint (keeping the second constant)
% % We can use some different values for the second joint: {30,60, 90,120}
% delta = (rhuman.kineconfig.joint_upperlimits(1)*180/pi - rhuman.kineconfig.joint_lowerlimits(1)*180/pi)/n_points
% for i= rhuman.kineconfig.joint_lowerlimits(1)*180/pi :delta: rhuman.kineconfig.joint_upperlimits(1)*180/pi 
%     
%     theta1=theta; 
%     theta1(1)=i*pi/180; 
%     j_geom_i=rhuman.getJacobGeom(theta1);
%     manip_i = j_geom_i(4:6,:)*j_geom_i(4:6,:)';
%     manipulabilities(cnt,:) = reshape(manip_i,1,9);
% 
%     % Normalize manipulability to volume = 1
%     eigs=eig(manip_i)
%     if eigs(1)<0 % matlab rounding errors -> -0.00000
%         eigs(1)=0
%     end
%     if eigs(2)<0
%         eigs(2)=0
%     end
%     if eigs(3)<0
%         eigs(3)=0
%     end
%     vol_i=prod(sqrt(eigs))*(4.0/3.0)*pi;
%     manip_i_normalized = scaleEllipsoidVolume(manip_i, 1/vol_i);                   % has to be 1
%     manipulabilities_normalized(cnt,:) = reshape(manip_i_normalized,1,9);
%     scales(cnt)= vol_i;
%     
%     rhuman.plot(theta1); 
%     
%     pause(0.3); 
% end

% Keeping the first constant = {0, 45, 90} VERTICAL
% Moving the second from 0 to 180

rhuman = rHuManModel('shoulderHeight',1.35,'verbose',true);
figure; 
theta=zeros(7,1); 
theta(1)=90*pi/180; %30
n_points=20;

rhuman.plot(theta); 
view(-79,22); 
axis([-0.6 0.6 -0.3 0.6 0.0 1.8]); 

manipulabilities=zeros(n_points, 9);
manipulabilities_normalized=zeros(n_points, 9); % all volume = 1
scales=zeros(1,n_points);
format long

% Rotation the first joint (keeping the second constant)
% Keeping the second joint: {30,60, 90,120}
delta=(rhuman.kineconfig.joint_upperlimits(2)*180/pi -79)/n_points
cnt=1
for i= 79 :delta: rhuman.kineconfig.joint_upperlimits(2)*180/pi 
    
    theta1=theta; 
    theta1(2)=i*pi/180; 
    
    j_geom_i=rhuman.getJacobGeom(theta1);
    manip_i = j_geom_i(4:6,:)*j_geom_i(4:6,:)';
    manipulabilities(cnt,:) = reshape(manip_i,1,9);

    % Normalize manipulability to volume = 1
    eigs=eig(manip_i)
    if eigs(1)<0 % matlab rounding errors -> -0.00000
        eigs(1)=0
    end
    if eigs(2)<0
        eigs(2)=0
    end
    if eigs(3)<0
        eigs(3)=0
    end
    vol_i=prod(sqrt(eigs))*(4.0/3.0)*pi;
    manip_i_normalized = scaleEllipsoidVolume(manip_i, 1/vol_i);                   % has to be 1
    manipulabilities_normalized(cnt,:) = reshape(manip_i_normalized,1,9);
    scales(cnt)= vol_i;
    
    rhuman.plot(theta1); 
    cnt=cnt+1
    pause(0.6); 
end

base_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/mapping"
dataset="sing_up_90"
dlmwrite(base_path+"/rhuman/"+dataset+"/manipulabilities_normalized.csv", manipulabilities_normalized, 'delimiter', ',', 'precision', 32);
csvwrite(base_path+"/rhuman/"+dataset+"/scales.csv", scales');
dlmwrite(base_path+"/rhuman/"+dataset+"/manipulabilities.csv", manipulabilities, 'delimiter', ',', 'precision', 64);



