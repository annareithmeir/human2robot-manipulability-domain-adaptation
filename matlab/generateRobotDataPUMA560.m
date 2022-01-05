% add rhuman to path
% First run 'startup_rvc' from the robotics toolbox

function [scales, positions] = generateRobotDataPUMA560(num, basePath)
     mdl_puma560;
     joints = rand(6,num) .* (p560.qlim(:,1) + (p560.qlim(:,2)-p560.qlim(:,1)));

     positions=zeros(num, 3);
     manipulabilities=zeros(num, 9);
     manipulabilities_normalized=zeros(num, 9); % all volume = 1
     scales=zeros(1,num);

     %fprintf(1,'\n');
     %disp('waittext(x,"fraction")')
	for i=1:num
		 %waittext(i/num,'fraction');
		disp("Step 1   "+ i+ "/"+ num);
		 joints_i = joints(:,i);
		 positions(i,:) = p560.fkine(joints_i).t';

                 % translational part of geomManip
		 j_geom_i=p560.jacob0(joints_i, 'trans');
                 j_geom_i=j_geom_i(:,1:3);
		 manip_i = j_geom_i*j_geom_i';
		 manipulabilities(i,:) = reshape(manip_i,1,9);

		% Normalize manipulability to volume = 1
		vol_i=prod(sqrt(eig(manip_i)))*(4.0/3.0)*pi;
		manip_i_normalized = scaleEllipsoidVolume(manip_i, 1/vol_i);
		%prod(sqrt(eig(manip_i_normalized)))*(4.0/3.0)*pi;                   % has to be 1
		manipulabilities_normalized(i,:) = reshape(manip_i_normalized,1,9);
		scales(i)= vol_i;
	end

	scales_normalized = (scales-min(scales))/(max(scales)-min(scales));     % volume in [0,1]

	csvwrite(basePath+"/r_manipulabilities_normalized.csv", manipulabilities_normalized);
	csvwrite(basePath+"/r_manipulabilities.csv", manipulabilities);
	csvwrite(basePath+"/r_scales.csv", scales');
	csvwrite(basePath+"/r_scales_normalized.csv", scales_normalized');
	csvwrite(basePath+"/r_positions.csv", positions);

end
