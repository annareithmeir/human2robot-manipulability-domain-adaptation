function prepare_data(tasks)

rhuman = rHuManModel('shoulderHeight',1.55,'verbose',true); % shoulderheight does not effect J
base_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/";

if ~exist(strcat(base_path+'data/', tasks), 'dir')
       mkdir(strcat(base_path+'data/', tasks))
end

for i=2:5 %loop over users  

    path=strcat(base_path,'dataExp');
    path=strcat(path, num2str(i));
    path= strcat(path, '_OpSimFKM_Kinematics.mat');
    data=load(path);
        
    for ll=4:size(data.exp_description,1) %loop over tasks (lines in description)       
        if contains(data.exp_description(ll,:), tasks)
            tasknumber= str2num(data.exp_description(ll,1:3))
            if ~ismember(tasknumber, data.INVALID_ExpTask) 

            
                outpath=strcat(base_path+'data/', tasks);
                outpath=strcat(outpath,'/exp', num2str(i)');
                outpath=strcat(outpath,'_task');
                outpath=strcat(outpath,num2str(tasknumber));

                s=data.expData_openSimFKM.task{tasknumber}.size;
                m = zeros(s,10);
                t=zeros(s,4);
                joints = zeros(s,8);

                time= data.expData_openSimFKM.task{1,tasknumber}.time;
                m(:,1)=time;
                t(:,1)=time;
                joints(:,1)=time;

                for jj=1:s
                    joints(jj,2:8)=data.expData_openSimFKM.task{tasknumber}.joints(:,jj)';
                     jtmp=rhuman.getJacobGeom(data.expData_openSimFKM.task{tasknumber}.joints(:,jj));
                     m(jj,2:10) = reshape(jtmp(1:3,:)*jtmp(1:3,:)', 1,9);
                     t(jj,2:4)= reshape(getPos(rhuman, data.expData_openSimFKM.task{tasknumber}.joints(:,jj)),1,3);
                end

                %csvwrite(strcat(outpath, '_m.csv'), m);
                %dlmwrite(strcat(outpath, '_t.csv'), t, 'delimiter', ',', 'precision', 64);
                dlmwrite(strcat(outpath, '_joints.csv'), joints, 'delimiter', ',', 'precision', 64);
            end
       end
    end
    
end

end