function createLookupTable(base_path)

%     positionsHuman = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_positions.csv");
%     scalesHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_scales_normalized.csv");
%     manipsHuman=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_manipulabilities_normalized.csv");
%     positionsRobot = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_positions.csv");
%     scalesRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_scales_normalized.csv");
%     manipsRobot=csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_manipulabilities_normalized.csv");
    
disp("creating lookup table...");
    manipsHuman=csvread(base_path+"/h_manipulabilities_normalized.csv");
    manipsRobot=csvread(base_path+"/r_manipulabilities_normalized.csv");
    scalesHuman=csvread(base_path+"/h_scales.csv");
    scalesRobot=csvread(base_path+"/r_scales.csv");
    

    %frob norm
    num= size(manipsHuman,1);
    frobs=zeros(num, num);
    for i=1:num
        disp("Step 1   "+ i+ "/"+ num);
        mh=reshape(manipsHuman(i,:),3,3);
        for j=1:num
            mr=reshape(manipsRobot(j,:),3,3);
            frobs(i,j)= norm(logm(mh^-.5*mr*mh^-.5),'fro');
        end
    end
    
    [minValuesFrob, minIndicesFrob] = min(frobs,[],2); % index array of closest index in robot data for each frob err of human
    
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
    w1=0.5;
    w2=0.5;
    errs = w1.*frobs + w2.* vol_errs;
    
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
    
    csvwrite(base_path+"/lookup_trafos_naive.csv", affine_trafos);

end