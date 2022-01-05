function createLookupTable(base_path_h, base_path_r)

%     positionsHuman = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_positions.csv");
%     scalesHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_scales_normalized.csv");
%     manipsHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_manipulabilities_normalized.csv");
%     positionsRobot = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_positions.csv");
%     scalesRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_scales_normalized.csv");
%     manipsRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_manipulabilities_normalized.csv");
    
    disp("creating lookup table...");

    source_name = split(base_path_h,'/');
    source_name=source_name(size(source_name,1));
    target_name = split(base_path_r,'/');
    target_name=target_name(size(target_name,1));

    if source_name == "human"
        manipsHuman=csvread(base_path_h+"/h_manipulabilities_normalized.csv");
	manipsRobot=csvread(base_path_r+"/r_manipulabilities_normalized.csv");
	scalesHuman=csvread(base_path_h+"/h_scales.csv");
	scalesRobot=csvread(base_path_r+"/r_scales.csv");
    else
    	manipsHuman=csvread(base_path_h+"/r_manipulabilities_normalized.csv");
    	manipsRobot=csvread(base_path_r+"/r_manipulabilities_normalized.csv");
    	scalesHuman=csvread(base_path_h+"/r_scales.csv");
    	scalesRobot=csvread(base_path_r+"/r_scales.csv");
    end
    

    %matrix error
    num= size(manipsHuman,1);
    dist_errs=zeros(num, num);
    for i=1:num
        disp("Step 1   "+ i+ "/"+ num);
        mh=reshape(manipsHuman(i,:),3,3);
        for j=1:num
            mr=reshape(manipsRobot(j,:),3,3);
            dist_errs(i,j)= distanceLogEuclidean(mh, mr);
        end
    end
    
    [minValuesDistErrs, minIndicesDistErrs] = min(dist_errs,[],2); % index array of closest index in robot data for each frob err of human
    
    %normalized volume error
    vol_errs=zeros(num, num);
    for i=1:num
        disp("Step 2   "+ i+ "/"+ num);
        vh=scalesHuman(i,1);
        for j=1:num
            vr=scalesRobot(j,1);
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
        mh=reshape(manipsHuman(i,:),3,3);
        mr=reshape(manipsRobot(minIndicesCombined(i),:),3,3);
        
        L1=logmap(mr, mh); % exp(mr) wrt mh
        affine_trafos(i,:) = reshape(L1, 1,9);
    end
    

    dlmwrite(base_path_r+"/lookup_trafos_naive_"+source_name+"_to_"+target_name+".csv", affine_trafos,'delimiter', ',', 'precision', 64); % robot is target, human is source

end
