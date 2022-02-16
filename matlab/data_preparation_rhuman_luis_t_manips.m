% generates interpolated manips and trajectories
% to be used after calling data_preparation_rhuman_luis.py on same task

function data_preparation_rhuman_luis_t_manips(tasks)

rhuman = rHuManModel('shoulderHeight',1.35,'verbose',true); % shoulderheight does not effect J
base_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/";

if ~exist(strcat(base_path+'data/', tasks), 'dir')
       mkdir(strcat(base_path+'data/', tasks))
end

interp_path = base_path + "data/"+ tasks+"/interpolated/"


files = dir (strcat(interp_path,'/*joints.csv'))
L = length (files);

for i=1:L
    files(i).name
    joints_interp = csvread(strcat(interp_path,'/',files(i).name),1,0);
    s=size(joints_interp,1);

    m = zeros(s,10);
    t=zeros(s,4);
    
    m(:,1)=joints_interp(:,1);
    t(:,1)=joints_interp(:,1);

    for jj=1:s
         jtmp=rhuman.getJacobGeom(joints_interp(jj,2:8));
         m(jj,2:10) = reshape(jtmp(4:6,:)*jtmp(4:6,:)', 1,9);
         %m(jj,2:10) = reshape(jtmp(1:3,:)*jtmp(1:3,:)', 1,9);
         t(jj,2:4)= reshape(getPos(rhuman, joints_interp(jj,2:8)),1,3);
    end

    outpath = split(files(i).name,'_');
    outpath= interp_path + join(outpath(1:size(outpath,1)-1),'_');
    dlmwrite(strcat(outpath, '_m.csv'), m, 'delimiter', ',', 'precision', 64);
    dlmwrite(strcat(outpath, '_t.csv'), t, 'delimiter', ',', 'precision', 64);

end

end