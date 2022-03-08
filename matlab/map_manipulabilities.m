function mapped_manipulabilities = map_manipulabilities(base_path, robot_teacher, robot_student, lookup_dataset, mapping_dataset)
%function mapped_manipulabilities = map_manipulabilities(experiment, user)
    format long

    %disp(experiment+" user "+user)
    %desired_manipulabilities = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/"+experiment+"/"+user+"/xhat.csv");
    
%     teacher_manipulabilities_random = csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/manipulabilities.csv");
    teacher_manipulabilities_random = csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/manipulabilities.csv");

    affine_trafos = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/rhuman/10000/lookup_trafos_naive_rhuman_to_panda.csv");
%     affine_trafos = csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/lookup_trafos_naive_"+robot_teacher+"_to_"+robot_student+".csv");
    mapping_manipulabilities = csvread(base_path+"/realworld/"+mapping_dataset+"/manipulabilities.csv");
    mapping_manipulabilities=mapping_manipulabilities(1:20:end,:);

%     mapping_manipulabilities = csvread(base_path+"/"+robot_teacher+"/"+mapping_dataset+"/manipulabilities.csv");
    size(mapping_manipulabilities)
    mapped_manipulabilities=zeros(size(mapping_manipulabilities, 1),9);
    
    mapping_manipulabilities= reshape(mapping_manipulabilities, size(mapping_manipulabilities, 1),3,3);
    teacher_manipulabilities_random= reshape(teacher_manipulabilities_random, size(teacher_manipulabilities_random, 1),3,3);
    

    num = size(mapping_manipulabilities, 1);
    tol = 1.e-6;
    tic
    for i=1:num
        disp(i+ "/"+ num);
        
        [v,w] = eig(reshape(mapping_manipulabilities(i,:,:),3,3));
        w(w<tol & w>-tol) = 0;
        if min(w)==0
            wd=diag(w);
            wd(wd==0)=0.0001;
            Mdesired=v*diag(wd)*v';
        else
            Mdesired=reshape(mapping_manipulabilities(i,:,:),3,3);
        end
        
        [v,w] = eig(reshape(teacher_manipulabilities_random(i,:,:),3,3));
        w(w<tol & w>-tol) = 0;
        if min(w)==0
            wd=diag(w);
            wd(wd==0)=0.0001;
            Mh=v*diag(wd)*v';
        else
            Mh=reshape(teacher_manipulabilities_random(i,:,:),3,3);
        end
        

        % normalize
        %Mdesired = reshape(mapping_manipulabilities(i,:,:),3,3);
        
        [M, scale] = normalize_manipulability(Mdesired);
        
        % find closest in random manipulabilities
        errs=zeros(size(teacher_manipulabilities_random, 1), 1);
        for j=1:size(teacher_manipulabilities_random, 1)
            %Mh = reshape(teacher_manipulabilities_random(j,:),3,3);
            errs(j,1) = distanceLogEuclidean(Mh,M);
        end
        
        [minMh, minIndex] = min(errs,[],1);
        [v,w] = eig(reshape(teacher_manipulabilities_random(minIndex,:,:),3,3));
        w(w<tol & w>-tol) = 0;
        if min(w)==0
            wd=diag(w);
            wd(wd==0)=0.0001;
            nearestMh=v*diag(wd)*v';
        else
            nearestMh = reshape(teacher_manipulabilities_random(minIndex,:),3,3);
        end
        
        
        % perform trafo
        L1 = reshape(affine_trafos(minIndex,:),3,3);
%         L1=round(L1,6);
%          issymmetric(L1)
        Ac = transp_operator(nearestMh, M);
%         Ac=round(Ac,6);
%         eig(Ac)
	    L1 = Ac * L1 * Ac';
%         L1=round(L1,6)
%          issymmetric(L1)
%         eig(L1)
%          eig(M)
        M = expmap(L1, M);

               
%          eig(M)
       
	    assert(min(eig(M)) >0);
        [M, ~] = normalize_manipulability(M);
        eig(M);
        
        % denormalize
        M = scaleEllipsoidVolume(M, scale);
        assert(min(eig(M))>0);
        
        %save
        mapped_manipulabilities(i,:) = reshape(M, 1,9);



%%%%%%%%
        
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
    
%     if ~exist(base_path+"/"+robot_student+"/"+mapping_dataset, 'dir')
%        mkdir(base_path+"/"+robot_student+"/"+mapping_dataset);
%     end
    toc
    dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/naive/"+mapping_dataset+"/manipulabilities_mapped_naive.csv", mapped_manipulabilities, 'delimiter', ',', 'precision', 64);
%     dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/naive/"+mapping_dataset+"/manipulabilities_interpolated_mapped_naive_"+lookup_dataset+".csv", mapped_manipulabilities, 'delimiter', ',', 'precision', 64);
%     dlmwrite(base_path+"/"+robot_student+"/"+mapping_dataset+"/manipulabilities_interpolated_mapped_naive.csv", mapped_manipulabilities, 'delimiter', ',', 'precision', 64);

end
