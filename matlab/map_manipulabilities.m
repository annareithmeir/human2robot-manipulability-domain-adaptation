function mapped_manipulabilities = map_manipulabilities(base_path, robot_teacher, robot_student, lookup_dataset, mapping_dataset)
%function mapped_manipulabilities = map_manipulabilities(experiment, user)

    %disp(experiment+" user "+user)
    %desired_manipulabilities = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/"+experiment+"/"+user+"/xhat.csv");
    
    teacher_manipulabilities_random = csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/manipulabilities.csv");
    affine_trafos = csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/lookup_trafos_naive_"+robot_teacher+"_to_"+robot_student+".csv");
    mapping_manipulabilities = csvread(base_path+"/"+robot_teacher+"/"+mapping_dataset+"/manipulabilities_interpolated.csv");

    mapping_manipulabilities = mapping_manipulabilities(2:size(mapping_manipulabilities,1),2:10);
    mapped_manipulabilities=zeros(size(mapping_manipulabilities, 1),9);
   
    % for each mapping manipulability
    num = size(mapping_manipulabilities, 1);
    
    for i=1:num
        disp(i+ "/"+ num);

        % normalize
        Mdesired = reshape(mapping_manipulabilities(i,:),3,3);
        [M, scale] = normalize_manipulability(Mdesired);
        
        % find closest in random manipulabilities
        errs=zeros(size(teacher_manipulabilities_random, 1), 1);
        for j=1:size(teacher_manipulabilities_random, 1)
            Mh = reshape(teacher_manipulabilities_random(j,:),3,3);
            errs(j,1) = distanceLogEuclidean(Mh,M);
        end
        
        [minMh, minIndex] = min(errs,[],1);
        nearestMh = reshape(teacher_manipulabilities_random(minIndex,:),3,3);

        
        % perform trafo
        L1 = reshape(affine_trafos(minIndex,:),3,3);
        Ac = transp_operator(nearestMh, M);
	    L1 = Ac * L1 * Ac';
        M = expmap(L1, M);

	    assert(min(eig(M)) >0);
        [M, ~] = normalize_manipulability(M);
        
        % denormalize
        M = scaleEllipsoidVolume(M, scale);
        assert(min(eig(M))>0);
        
        %save
        mapped_manipulabilities(i,:) = reshape(M, 1,9);
        
        % calculate error between desired and mapped
%         err_fro= norm(logm(Mdesired^-.5*M*Mdesired^-.5),'fro');
%         mdiff_i=logmap(Mdesired, M);          
%         err_norm=norm(symmat2vec(mdiff_i));
    
    end
    
%     if ~exist("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment, 'dir')
%        mkdir("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment);
%     end
%     
%     if ~exist("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user, 'dir')
%        mkdir("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user);
%     end
    
    %csvwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user+"/mapped_manipulabilities.csv", mapped_manipulabilities);
    
    if ~exist(base_path+"/"+robot_student+"/"+mapping_dataset, 'dir')
       mkdir(base_path+"/"+robot_student+"/"+mapping_dataset);
    end

    dlmwrite(base_path+"/"+robot_student+"/"+mapping_dataset+"/manipulabilities_interpolated_mapped_naive.csv", mapped_manipulabilities, 'delimiter', ',', 'precision', 64);

end
