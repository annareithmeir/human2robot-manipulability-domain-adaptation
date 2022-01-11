function mapped_manipulabilities = map_manipulabilities(base_path_h, base_path_r, desired_manipulabilities_path)
%function mapped_manipulabilities = map_manipulabilities(experiment, user)

    %disp(experiment+" user "+user)
    %desired_manipulabilities = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/"+experiment+"/"+user+"/xhat.csv");
    

    source_name = split(base_path_h,'/');
    source_name=source_name(size(source_name,1))
    target_name = split(base_path_r,'/');
    target_name=target_name(size(target_name,1))
    
    if source_name == "human"
        desired_manipulabilities = csvread(desired_manipulabilities_path);
        human_manipulabilities_random = csvread(base_path_h+"/h_manipulabilities.csv");
    else
        desired_manipulabilities = csvread(desired_manipulabilities_path);
        human_manipulabilities_random = csvread(base_path_h+"/r_manipulabilities.csv");
    end
    
    desired_manipulabilities = desired_manipulabilities(2:size(desired_manipulabilities,1),2:10);
    
    affine_trafos = csvread(base_path_r+"/lookup_trafos_naive_"+source_name+"_to_"+target_name+".csv");
    %affine_trafos = csvread(base_path_r+"/lookup_trafos_naive_"+source_name+"_to_"+target_name+".csv");
   
    mapped_manipulabilities=zeros(size(desired_manipulabilities, 1),9);
   
    % for each desired manipulability
    num = size(desired_manipulabilities, 1);
    
    for i=1:num
        disp(i+ "/"+ num);

        % normalize
        Mdesired = reshape(desired_manipulabilities(i,:),3,3);
        [M, scale] = normalize_manipulability(Mdesired);
        
        % find closest in random manipulabilities
        errs=zeros(size(human_manipulabilities_random, 1), 1);
        for j=1:size(human_manipulabilities_random, 1)
            Mh = reshape(human_manipulabilities_random(j,:),3,3);
            errs(j,1) = distanceLogEuclidean(Mh,M);
        end
        
        errs;
        
        [minMh, minIndex] = min(errs,[],1)
        nearestMh = reshape(human_manipulabilities_random(minIndex,:),3,3)

        
        % perform trafo
        L1 = reshape(affine_trafos(minIndex,:),3,3);
        % M= expmap(L1, nearestMh);

        Ac = transp_operator(nearestMh, M)

	L1
	L1 = Ac * L1 * Ac'
        M= expmap(L1, M)

	assert(min(eig(M)) >0)
        [M, ~] = normalize_manipulability(M)
	disp("---")
        
        % denormalize
        M = scaleEllipsoidVolume(M, scale)
        assert(min(eig(M))>0)
        
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

    source_name = split(base_path_r,'/');
    base_path= join(source_name(1:size(source_name,1)-2),'/');
    source_name = source_name(size(source_name,1));
    
    if ~exist(base_path+"/results/"+source_name, 'dir')
       mkdir(base_path+"/results/"+source_name);
    end

    dlmwrite(base_path+"/results/"+source_name+"/mapped_manipulabilities_human_naive.csv", mapped_manipulabilities, 'delimiter', ',', 'precision', 64);

end
