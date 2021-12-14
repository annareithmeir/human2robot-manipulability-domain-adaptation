% add rhuman to path

function [scales, positions] = generateHumanData(shoulderHeight, num, basePath)
     rhuman = rHuManModel('shoulderHeight',shoulderHeight,'verbose',true);
     joints=rhuman.getRandJoints('length',num, 'seed', 1);

     positions=zeros(num, 3);
     manipulabilities=zeros(num, 9);
     manipulabilities_normalized=zeros(num, 9); % all volume =1
     scales=zeros(1,num);

     %fprintf(1,'\n');
     %disp('waittext(x,"fraction")')
	for i=1:num
		 %waittext(i/num,'fraction');
		disp("Step 1   "+ i+ "/"+ num);
		 joints_i = joints(:,i);
		 positions(i,:) = rhuman.getPos(joints_i);

		 j_geom_i=rhuman.getJacobGeom(joints_i);
		 manip_i = j_geom_i(4:6,:)*j_geom_i(4:6,:)';
		 manipulabilities(i,:) = reshape(manip_i,1,9);

		% Normalize manipulability to volume = 1
		vol_i=prod(sqrt(eig(manip_i)))*(4.0/3.0)*pi;
		manip_i_normalized = scaleEllipsoidVolume(manip_i, 1/vol_i);
		%prod(sqrt(eig(manip_i_normalized)))*(4.0/3.0)*pi;                   % has to be 1
		manipulabilities_normalized(i,:) = reshape(manip_i_normalized,1,9);
		scales(i)= vol_i;
	end

	scales_normalized = (scales-min(scales))/(max(scales)-min(scales));     % volume in [0,1]

	csvwrite(basePath+"/h_manipulabilities_normalized.csv", manipulabilities_normalized);
	csvwrite(basePath+"/h_manipulabilities.csv", manipulabilities);
	csvwrite(basePath+"/h_scales.csv", scales');
	csvwrite(basePath+"/h_scales_normalized.csv", scales_normalized');
	csvwrite(basePath+"/h_positions.csv", positions);

end
