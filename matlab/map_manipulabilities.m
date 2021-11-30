function mapped_manipulabilities = map_manipulabilities(experiment, user)
    disp(experiment+" user "+user)
    desired_manipulabilities = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/"+experiment+"/"+user+"/xhat.csv");
    %desired_manipulabilities = csvread("/home/nnrthmr/Desktop/someData.csv");
    affine_trafos = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/lookup_trafos_log_exp.csv");
    human_manipulabilities_random = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_manipulabilities_normalized.csv");
    
    mapped_manipulabilities=zeros(size(desired_manipulabilities, 1),9);
    %human_manipulabilities_random = human_manipulabilities_random(1:5000,:);
   
    % for each desired manipulability
    disp(size(desired_manipulabilities,1))
    
    for i=1:size(desired_manipulabilities, 1)
    %for i=1:190
        
        
        % normalize
        Mdesired = reshape(desired_manipulabilities(i,:),3,3);
        [M, scale] = normalize_manipulability(Mdesired);
        
        % find closest in random manipulabilities
        errs=zeros(size(human_manipulabilities_random, 1), 1);
        for j=1:size(human_manipulabilities_random, 1)
            Mh = reshape(human_manipulabilities_random(j,:),3,3);
            errs(j,1) = norm(logm(Mh^-.5*M*Mh^-.5),'fro');
        end
        
        errs;
        
        [minMh, minIndex] = min(errs,[],1);
        nearestMh = reshape(human_manipulabilities_random(minIndex,:),3,3);
        nearestMh
        
        % perform trafo
        L1 = reshape(affine_trafos(minIndex,:),3,3);
        %M = A*M
        M= expmap(L1, nearestMh);
        
        % denormalize
        M = scaleEllipsoidVolume(M, scale);
        assert(min(eig(M))>=0);
        M;
        
        %save
        mapped_manipulabilities(i,:) = reshape(M, 1,9);
        
        % calculate error between desired and mapped
        err_fro= norm(logm(Mdesired^-.5*M*Mdesired^-.5),'fro')
        mdiff_i=logmap(Mdesired, M);          
        err_norm=norm(symmat2vec(mdiff_i))
    
    end
    
    if ~exist("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment, 'dir')
       mkdir("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment);
    end
    
    if ~exist("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user, 'dir')
       mkdir("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user);
    end
    
    %csvwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user+"/mapped_manipulabilities.csv", mapped_manipulabilities);
    csvwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/"+experiment+"/"+user+"/mapped_manipulabilities_log_exp.csv", mapped_manipulabilities);
    %csvwrite("/home/nnrthmr/Desktop/someDataMapped_log_exp.csv", mapped_manipulabilities);

end