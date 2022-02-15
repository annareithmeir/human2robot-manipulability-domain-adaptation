function cpd8d(base_path, robot_teacher, robot_student, lookup_dataset, mapping_dataset)
    %clear all; close all; clc;
    addpath "/home/nnrthmr/CLionProjects/ma_thesis/matlab/CPD/github/CoherentPointDrift-master/core/";
    addpath(genpath("/home/nnrthmr/CLionProjects/ma_thesis/matlab/CPD/github/CoherentPointDrift-master"));
    
    % find transformation based on lookup datatset
    disp(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/8d/manipulabilities.csv");
    disp(base_path+"/"+robot_student+"/"+lookup_dataset+"/8d/manipulabilities.csv");
    
    Y = csvread(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/8d/manipulabilities.csv");
    X = csvread(base_path+"/"+robot_student+"/"+lookup_dataset+"/8d/manipulabilities.csv");
    %X=X(1:10:end,:);
    %Y=Y(1:10:end,:);
    disp("Number of samples used in CPD: ")
    size(X,1)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    opt.method='affine'; % use rigid registration
    %opt.viz=1;          % show every iteration
    opt.outliers=0.001;     % do not assume any noise 

    opt.normalize=1;    % normalize to unit variance and zero mean before registering (default)
    opt.scale=0;        % estimate global scaling too (default)
    opt.rot=1;          % estimate strictly rotational matrix (default)
    opt.corresp=0;      % do not compute the correspondence vector at the end of registration (default)

    opt.max_it=10000;     % max number of iterations
    opt.tol=1e-30;       % tolerance
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     opt.method='nonrigid'; % use nonrigid registration
%     opt.beta=2;            % the width of Gaussian kernel (smoothness)
%     opt.lambda=8;          % regularization weight
% 
%     opt.viz=1;              % show every iteration
%     opt.outliers=0.7;       % use 0.7 noise weight
%     opt.fgt=0;              % do not use FGT (default)
%     opt.normalize=1;        % normalize to unit variance and zero mean before registering (default)
%     opt.corresp=1;          % compute correspondence vector at the end of registration (not being estimated by default)
% 
%     opt.max_it=100;         % max number of iterations
%     opt.tol=1e-10;          % tolerance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % registering Y to X
    [Transform, Correspondence]=cpd_register(X,Y,opt);

%     dlmwrite(base_path+"/"+robot_student+"/"+lookup_dataset+"/8d/manipulabilities_mapped.csv", Transform.Y,'delimiter', ',', 'precision', 64);
%     dlmwrite(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/8d/s_cpd_panda_to_toy_data.txt", Transform.s,'delimiter', ',', 'precision', 64);
%     dlmwrite(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/8d/R_cpd_panda_to_toy_data.txt", Transform.R,'delimiter', ',', 'precision', 64);
%     dlmwrite(base_path+"/"+robot_teacher+"/"+lookup_dataset+"/8d/t_cpd_panda_to_toy_data.txt", Transform.t,'delimiter', ',', 'precision', 64);
    
    
    % apply transformation found before to map_dataset
    if(robot_teacher=="rhuman")
        Y_new = csvread(base_path+"/"+robot_teacher+"/"+mapping_dataset+"/8d/manipulabilities.csv");
    else
        Y_new = csvread(base_path+"/"+robot_teacher+"/"+mapping_dataset+"/8d/manipulabilities_interpolated.csv");
    end
    
    
    %rigid/affine
    Y_new_mapped=Transform.s*Y_new*Transform.R'+repmat(Transform.t',[size(Y_new,1) 1]);
    
    %nonrigid
%     G=cpd_G(Y_new,Y_new,opt.beta)
%     Transform.W
%     Y_new_mapped=Y_new+G*Transform.W;
    if(robot_teacher=="rhuman")
        dlmwrite(base_path+"/"+robot_student+"/"+mapping_dataset+"/8d/manipulabilities_mapped_cpd.csv", Y_new_mapped,'delimiter', ',', 'precision', 64);
    else
        dlmwrite(base_path+"/"+robot_student+"/"+mapping_dataset+"/8d/manipulabilities_interpolated_mapped_cpd.csv", Y_new_mapped,'delimiter', ',', 'precision', 64);
    end
end