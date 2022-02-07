clear all; close all; clc;

data_franka = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/8d/manipulabilities.csv");
data_toy = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/100/8d/manipulabilities.csv");

X=data_toy;
Y=data_franka;

% add a random rigid transformation
%R=cpd_R(rand(1),rand(1),rand(1));
%X=rand(1)*X*R'+1;

% Set the options
opt.method='affine'; % use rigid registration
%opt.viz=1;          % show every iteration
opt.outliers=0;     % do not assume any noise 

opt.normalize=1;    % normalize to unit variance and zero mean before registering (default)
opt.scale=1;        % estimate global scaling too (default)
opt.rot=1;          % estimate strictly rotational matrix (default)
opt.corresp=0;      % do not compute the correspondence vector at the end of registration (default)

opt.max_it=1000;     % max number of iterations
opt.tol=1e-8;       % tolerance


% registering Y to X
[Transform, Correspondence]=cpd_register(X,Y,opt);

dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/100/8d/manipulabilities_mapped.csv", Transform.Y,'delimiter', ',', 'precision', 64);
dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/8d/s_cpd_panda_to_toy_data.txt", Transform.s,'delimiter', ',', 'precision', 64);
dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/8d/R_cpd_panda_to_toy_data.txt", Transform.R,'delimiter', ',', 'precision', 64);
dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/8d/t_cpd_panda_to_toy_data.txt", Transform.t,'delimiter', ',', 'precision', 64);

%figure,cpd_plot_iter(X, Y); title('Before');
%figure,cpd_plot_iter(X, Transform.Y);  title('After registering Y to X');