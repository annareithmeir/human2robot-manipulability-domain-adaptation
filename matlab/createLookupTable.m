function createLookupTable(base_path, robot_teacher, robot_student, lookup_dataset)

%     positionsHuman = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_positions.csv");
%     scalesHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_scales_normalized.csv");
%     manipsHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_manipulabilities_normalized.csv");
%     positionsRobot = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_positions.csv");
%     scalesRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_scales_normalized.csv");
%     manipsRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_manipulabilities_normalized.csv");
    
    disp("creating lookup table...");

    manips_robot_teacher=csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/manipulabilities_normalized.csv");
    manips_robot_student=csvread(base_path+"/"+robot_student+"/"+lookup_dataset+"/manipulabilities_normalized.csv");
    scales_robot_teacher=csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/scales.csv");
    scales_robot_student=csvread(base_path+"/"+robot_student+"/"+lookup_dataset+"/scales.csv");

    %matrix error
    num= size(manips_robot_teacher,1);
    dist_errs=zeros(num, num);
    for i=1:num
        disp("Step 1   "+ i+ "/"+ num);
        mh=reshape(manips_robot_teacher(i,:),3,3);
        for j=1:num
            mr=reshape(manips_robot_student(j,:),3,3);
            dist_errs(i,j)= distanceLogEuclidean(mh, mr);
        end
    end
    
    [minValuesDistErrs, minIndicesDistErrs] = min(dist_errs,[],2); % index array of closest index in robot data for each frob err of human
    
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
    
    [minValuesFrob, minIndicesFrob] = min(vol_errs,[],2); % index array of closest index in robot data for each vol err of human
    
    
    % combined errors
    w1=1;
    w2=0;
    errs = w1.*dist_errs + w2.* vol_errs;
    
    [minValuesCombined, minIndicesCombined] = min(errs,[],2); % index array of closest index in robot data for each  err of human
    
    % affine trafos
    affine_trafos=zeros(num,9);
    for i=1:num
        disp("Step 3   "+ i+ "/"+ num);
        mh=reshape(manips_robot_teacher(i,:),3,3);
        mr=reshape(manips_robot_student(minIndicesCombined(i),:),3,3);
        
        L1=logmap(mr, mh); % exp(mr) wrt mh
        affine_trafos(i,:) = reshape(L1, 1,9);
    end
    

    dlmwrite(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/lookup_trafos_naive_"+robot_teacher+"_to_"+robot_student+".csv", affine_trafos,'delimiter', ',', 'precision', 64); % robot is target, human is source

end
