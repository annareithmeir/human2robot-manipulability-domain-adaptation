function get_data_statistics(base_path, robot1, robot2)

	if robot1 == "human"
		manipsHuman=csvread(base_path+"/data/human/h_manipulabilities_normalized.csv");
		manipsRobot=csvread(base_path+"/data/"+robot2+"/r_manipulabilities_normalized.csv");
	else
		manipsHuman=csvread(base_path+"/data/"+robot1+"/r_manipulabilities_normalized.csv");
		manipsRobot=csvread(base_path+"/data/"+robot2+"/r_manipulabilities_normalized.csv");
	end

	% kercher means
	manipsHuman = reshape(manipsHuman', 3,3, []);
	manipsRobot = reshape(manipsRobot', 3,3, []);
	mean1 = spdMean(manipsHuman) %dxdxn
	mean2 = spdMean(manipsRobot) %dxdxn

	% mean distances to mean
	mean_dist1 = 0;
	mean_dist2 = 0;
	for i=1:size(manipsHuman,3)
		mean_dist1=mean_dist1 + distanceLogEuclidean(mean1, manipsHuman(:,:,i));
		mean_dist2=mean_dist2 + distanceLogEuclidean(mean2, manipsRobot(:,:,i));
	end
	mean_dist1 = mean_dist1 / size(manipsHuman,3)
	mean_dist2 = mean_dist2 / size(manipsHuman,3)

	% distance of the means
	dist = distanceLogEuclidean(mean1, mean2)

	fid = fopen(base_path+'/data_statistics.txt','w');
	fprintf(fid, 'Distances between means : %.3f\nMean distance to mean robot1 : %.3f\nMean distance to mean robot2: %.3f', dist, mean_dist1, mean_dist2);

	fclose(fid);

end
