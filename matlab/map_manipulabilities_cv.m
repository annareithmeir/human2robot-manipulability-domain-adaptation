function mapped_manipulabilities = map_manipulabilities(base_path, robot_teacher, robot_student, dataset, cv_k)
%function mapped_manipulabilities = map_manipulabilities(experiment, user)

    %disp(experiment+" user "+user)
    %desired_manipulabilities = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/"+experiment+"/"+user+"/xhat.csv");
    
    teacher_manipulabilities_random = csvread(base_path+"/"+robot_teacher+"/"+dataset+"/manipulabilities.csv");
    affine_trafos = csvread(base_path+"/"+robot_teacher+"/"+dataset+"/cv/lookup_trafos_naive_"+robot_teacher+"_to_"+robot_student+".csv");
    
    tmp=teacher_manipulabilities_random;
    cv_idx=csvread(base_path+"/"+robot_student+"/"+dataset+"/cv/cv_idx.csv");
    teacher_manipulabilities_random=tmp(cv_idx(cv_k,:)==0,:);

    mapping_manipulabilities = tmp(cv_idx(cv_k,:)==1,:);
    mapped_manipulabilities=zeros(size(mapping_manipulabilities, 1),9);
   
    % for each mapping manipulability
    num = size(mapping_manipulabilities, 1);
    
    for i=1:num
        disp(i+ "/"+ num);

        % normalize
        M = reshape(mapping_manipulabilities(i,:),3,3);
        [M, scale] = normalize_manipulability(M);
        
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
    end
    
%     if ~exist("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment, 'dir')
%        mkdir("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment);
%     end
%     
%     if ~exist("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user, 'dir')
%        mkdir("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user);
%     end
    
    %csvwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user+"/mapped_manipulabilities.csv", mapped_manipulabilities);

    dlmwrite(base_path+"/"+robot_student+"/"+dataset+"/cv/manipulabilities_mapped_naive.csv", mapped_manipulabilities, 'delimiter', ',', 'precision', 64);

end
