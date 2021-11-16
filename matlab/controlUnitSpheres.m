% add rhuman to path

num = 3;
iter=1000;
dt=1e-2;
km0=0.005;
rhuman = rHuManModel('shoulderHeight',1.35,'verbose',true);
joints=rhuman.getRandJoints('length',num);

errs=zeros(num, iter);
positions=zeros(num, 3);
manipulabilities=zeros(num,9);

figure
hold on

for i=1:num
    joints_i = joints(:,i);
    km=km0;
    scaled=0;
    manip_desired=eye(3);
    
    %rhuman.plot(joints_i);
    %view(-70,22);
    %axis([-0.6 0.6 -0.3 0.6 0.0 1.8]);
    %pause(1);
    
    for t=1:iter
        
        j_geom_i=rhuman.getJacobGeom(joints_i);
        manip_i = j_geom_i(1:3,:)*j_geom_i(1:3,:)';
        
        manip_jacob_i = compute_red_manipulability_Jacobian(j_geom_i, 1:3);
        %manip_jacob_i(1:3,:) = 0;
        
        mdiff_i=logmap(manip_desired, manip_i);
        dqt1 = pinv(manip_jacob_i)*km*symmat2vec(mdiff_i);
        
        joints_i= joints_i + dqt1*dt;
        
        err=norm(logm(manip_desired^-.5*manip_i*manip_desired^-.5),'fro')
        
        errs(i,t)=err;
        
        if err < 1.2 && scaled==0
            manip_desired = trace(manip_i)/3 * eye(3);
            scaled=1;
        end
        
        %km=km0*err;
        
        %rhuman.plot(joints_i);
        %view(-70,22);
        %axis([-0.6 0.6 -0.3 0.6 0.0 1.8]);
        %pause(1);
        
        
        
    end
    
    manipulabilities(i,:)= reshape(manip_i,1,9);
    positions(i,:) = rhuman.getPos(joints_i);
    
    x = 1:size(errs,2) ; 
    
    for k = 1:size(errs,1)
        plot(x,errs(k,:))
    end
    
    %t_i = rhuman.getPos(joints_i);
    %x = rhuman.getFKM(joints_i);
    
end