% add rhuman to path
% First run 'startup_rvc' from the robotics toolbox

function generateRobotDataToy(basePath)
     manipulabilities = csvread(basePath+"/toy_data/r_manipulabilities.csv");
     num = size(manipulabilities,1);
     manipulabilities_normalized=zeros(num, 9); % all volume = 1
     scales=zeros(1,num);

	for i=1:num
		disp("Step 1   "+ i+ "/"+ num);
		manip_i = reshape(manipulabilities(i,:),3,3);

		if min(eig(manip_i)) <= 0
			disp("_________________________________________________________________________________")
		end

		% Normalize manipulability to volume = 1
		vol_i=prod(sqrt(eig(manip_i)))*(4.0/3.0)*pi;
		manip_i_normalized = scaleEllipsoidVolume(manip_i, 1/vol_i);
		%prod(sqrt(eig(manip_i_normalized)))*(4.0/3.0)*pi;                   % has to be 1
		manipulabilities_normalized(i,:) = reshape(manip_i_normalized,1,9);
		scales(i)= vol_i;
	end

	scales_normalized = (scales-min(scales))/(max(scales)-min(scales));     % volume in [0,1]

	dlmwrite(basePath+"/toy_data/r_manipulabilities_normalized.csv", manipulabilities_normalized, 'delimiter', ',', 'precision', 32);
	%dlmwrite(basePath+"/r_manipulabilities.csv", manipulabilities, 'delimiter', ',', 'precision', 32);
	csvwrite(basePath+"/toy_data/r_scales.csv", scales');
	csvwrite(basePath+"/toy_data/r_scales_normalized.csv", scales_normalized');
	%csvwrite(basePath+"/r_positions.csv", positions);

end