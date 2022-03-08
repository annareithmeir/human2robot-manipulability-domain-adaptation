function createLookupTable(base_path, robot_teacher, robot_student, lookup_dataset, cv_k)

%     positionsHuman = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_positions.csv");
%     scalesHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_scales_normalized.csv");
%     manipsHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_manipulabilities_normalized.csv");
%     positionsRobot = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_positions.csv");
%     scalesRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_scales_normalized.csv");
%     manipsRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_manipulabilities_normalized.csv");
    
    disp("creating lookup table...");

    manips_robot_teacher=csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/manipulabilities_normalized_eig.csv");
    manips_robot_student=csvread(base_path+"/"+robot_student+"/"+lookup_dataset+"/manipulabilities_normalized_eig.csv");
    %manips_robot_teacher=csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/manipulabilities.csv");
    %manips_robot_student=csvread(base_path+"/"+robot_student+"/"+lookup_dataset+"/manipulabilities.csv");
    scales_robot_teacher=csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/scales.csv");
    scales_robot_student=csvread(base_path+"/"+robot_student+"/"+lookup_dataset+"/scales.csv");
    
    if nargin ==5
        cv_idx=csvread(base_path+"/"+robot_student+"/"+lookup_dataset+"/cv/cv_idx.csv");
        manips_robot_teacher=manips_robot_teacher(cv_idx(cv_k,:)==0,:);
        manips_robot_student=manips_robot_student(cv_idx(cv_k,:)==0,:);
        scales_robot_teacher=scales_robot_teacher(cv_idx(cv_k,:)==0,:);
        scales_robot_student=scales_robot_student(cv_idx(cv_k,:)==0,:);
    end

    tic
    %matrix error
    assert(size(manips_robot_teacher,1) == size(manips_robot_student,1));
    
    
    num= size(manips_robot_teacher,1);
    dist_errs=zeros(num, num);
    for i=1:num
        disp("Step 1   "+ i+ "/"+ num);
        mh=reshape(manips_robot_teacher(i,:),3,3);
%         [v,w] = eig(mh);
%         tol = 1.e-6;
%         w(w<tol & w>-tol) = 0;
%         if min(w)==0
%             wd=diag(w);
%             wd(wd==0)=0.0001;
%             mh=v*diag(wd)*v';
%         end
        
        for j=1:num
            mr=reshape(manips_robot_student(j,:),3,3);
%             [v,w] = eig(mr);
%             tol = 1.e-6;
%             w(w<tol & w>-tol) = 0;
%             if min(w)==0
%                 wd=diag(w);
%                 wd(wd==0)=0.0001;
%                 mr=v*diag(wd)*v';
%             end
            dist_errs(i,j)= distanceLogEuclidean(mh, mr);
        end
    end
    
    dist_errs
    
    %[minValuesDistErrs, minIndicesDistErrs] = min(dist_errs,[],2); % index array of closest index in robot data for each frob err of human
    
    %normalized volume error
    vol_errs=zeros(num, num);
    for i=1:num
        disp("Step 2   "+ i+ "/"+ num);
        vh=scales_robot_teacher(i,1);
        for j=1:num
            vr=scales_robot_student(j,1);
            vol_errs(i,j)= norm(vr-vh);
        end
    end
    
    vol_errs
    
    %[minValuesFrob, minIndicesFrob] = min(vol_errs,[],2); % index array of closest index in robot data for each vol err of human
    
    
    % combined errors
    w1=0.5;
    w2=0.5;
    errs = w1.*dist_errs + w2.* vol_errs
    
    [minValuesCombined, minIndicesCombined] = min(errs,[],2); % index array of closest index in robot data for each  err of human
    
    % affine trafos
    affine_trafos=zeros(num,9);
    for i=1:num
        disp("Step 3   "+ i+ "/"+ num);
%         mh=reshape(manips_robot_teacher(i,:),3,3);
        mr=reshape(manips_robot_student(minIndicesCombined(i),:),3,3);
%         [v,w] = eig(mr);
%         tol = 1.e-6;
%         w(w<tol & w>-tol) = 0;
%         if min(w)==0
%             wd=diag(w);
%             wd(wd==0)=0.0001;
%             mr=v*diag(wd)*v';
%         end
        
        mh=reshape(manips_robot_teacher(i,:),3,3);
%         [v,w] = eig(mh);
%         tol = 1.e-6;
%         w(w<tol & w>-tol) = 0;
%         if min(w)==0
%             wd=diag(w);
%             wd(wd==0)=0.0001;
%             mh=v*diag(wd)*v';
%         end
        
        L1=logmap(mr, mh); % exp(mr) wrt mh
        affine_trafos(i,:) = reshape(L1, 1,9);
    end
    
    toc
    
%     if nargin == 5
%         dlmwrite(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/cv/lookup_trafos_naive_"+robot_teacher+"_to_"+robot_student+".csv", affine_trafos,'delimiter', ',', 'precision', 64); % robot is target, human is source
%     else
        dlmwrite(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/lookup_trafos_naive_"+robot_teacher+"_to_"+robot_student+".csv", affine_trafos,'delimiter', ',', 'precision', 64); % robot is target, human is source
%     end
end
