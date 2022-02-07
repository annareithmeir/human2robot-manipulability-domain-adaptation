% add rhuman to path

%default shoulder height=1.35
function [scales, positions] = collectAndNormalizeHumanArm(shoulderHeight)
    num=20;
    rhuman = rHuManModel('shoulderHeight',shoulderHeight,'verbose',true);
    joints=rhuman.getRandJoints('length',num, 'seed', 1);

    positions=zeros(num, 3);
    manipulabilities=zeros(num, 9);
    manipulabilities_normalized=zeros(num, 9); % all volume =1
    scales=zeros(1,num);
% 
%     manipulabilities=csvread("/home/nnrthmr/PycharmProjects/ma_thesis/data/h_manipulabilities.csv");
%     num=size(manipulabilities,1)


    for i=1:num
        i
        joints_i = joints(:,i);
        positions(i,:) = rhuman.getPos(joints_i);
        
        % Compute manipulability
        j_geom_i=rhuman.getJacobGeom(joints_i);
        manip_i = j_geom_i(4:6,:)*j_geom_i(4:6,:)';
        manipulabilities(i,:) = reshape(manip_i,1,9);

        manip_i = reshape(manipulabilities(i,:),3,3);
        
        % Normalize manipulability to volume = 1
        vol_i=prod(sqrt(eig(manip_i)))*(4.0/3.0)*pi;
        manip_i_normalized = scaleEllipsoidVolume(manip_i, 1/vol_i);
        %prod(sqrt(eig(manip_i_normalized)))*(4.0/3.0)*pi;                   % has to be 1
        manipulabilities_normalized(i,:) = reshape(manip_i_normalized,1,9);
        scales(i)= vol_i;
    end

    scales_normalized = (scales-min(scales))/(max(scales)-min(scales));     % volume in [0,1]
    
    csvwrite("/home/nnrthmr/PycharmProjects/ma_thesis/data/h_manipulabilities_normalized_2.csv", manipulabilities_normalized);
    csvwrite("/home/nnrthmr/PycharmProjects/ma_thesis/data/h_manipulabilities_2.csv", manipulabilities);
    %csvwrite("/home/nnrthmr/PycharmProjects/ma_thesis/data/h_scales_2.csv", scales');
    %csvwrite("/home/nnrthmr/PycharmProjects/ma_thesis/data/h_scales_normalized.csv", scales_normalized');
    %csvwrite("/home/nnrthmr/PycharmProjects/ma_thesis/data/h_positions.csv", positions);
    
end
