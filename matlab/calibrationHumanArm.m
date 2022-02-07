% add rhuman to path

%default shoulder height=1.35
function [scales, positions] = calibrationHumanArm(num, shoulderHeight)
    iter=100;
    dt=1e-2;
    km=1.5; % gain
    eps=0.0005; %singularity region
    damping_max=0.0002; % max damping factor
    
    rhuman = rHuManModel('shoulderHeight',shoulderHeight,'verbose',true);
    joints=rhuman.getRandJoints('length',num, 'seed', 1);

    errs=zeros(num, iter);
    positions=zeros(num, 3);
    scales=zeros(1,num);

    figure
    hold on

    for i=1:num
        maniplist=[];
        joints_i = joints(:,i);
        manip_desired=eye(3);
        %manip_desired(1,1)=0.1;
%         manip_desired=[0.0823 ,  -0.0180   , 0.0050;
%                         -0.0180,    0.0040  , -0.0011;
%                         0.0050 ,  -0.0011 ,   0.0024];

        for t=1:iter
            j_geom_i=rhuman.getJacobGeom(joints_i);
            j_geom_i_tmp=j_geom_i;
            manip_i = j_geom_i_tmp(4:6,:)*j_geom_i_tmp(4:6,:)';
            manip_jacob_i = compute_red_manipulability_Jacobian(j_geom_i_tmp, 4:6);  %in vector form
            
            if t==1
                'before'
                manip_i
            end
            
            %normalization (scale sphere to same volume as current manipulability)
            if t==1
            %if mod(t,100)==0
                S =  eig(manip_i);
                ve= prod(sqrt(S))*(4.0/3.0)*pi;
                %manip_desired = eye(3) * nthroot(ve/(4.0/3.0*pi),3)^2;
                
                manip_desired = eye(3) * nthroot(ve/(4.0/3.0*pi),3)^2; % singularity in one direction
                %manip_desired(1,1)=1e-6;
                
                Sd =  eig(manip_desired);
                vd= prod(sqrt(Sd))*(4.0/3.0)*pi; %vd must be equal to ve
            end
            
            
            mdiff_i=logmap(manip_desired, manip_i);          
            errs(i,t)=norm(symmat2vec(mdiff_i));
            err=norm(logm(manip_desired^-.5*manip_i*manip_desired^-.5),'fro');
            
            % damping
            [U,S,~] =  svd(manip_jacob_i);
            rankJ = rank(manip_jacob_i);
            S = S(1:rankJ,1:rankJ);
            min_sing_val = S(rankJ,rankJ); 
            Um=[];

            if min_sing_val <= eps
                for ii = rankJ:-1:1                 
                    if S(ii,ii) <= eps
                        Um = [U(:,ii) Um];
                    end
                end
                damping_factor = (1-(min_sing_val/eps)^2)*damping_max;
            else
                damping_factor = 0;
            end
            
            if(size(Um,1)==0)
               Um=zeros(6,6); 
            end
                       
         
            %manip_invmatrix =  manip_jacob_i'*pinv(manip_jacob_i*manip_jacob_i' + 0.00001*eye(6) );
            manip_invmatrix =  manip_jacob_i'*pinv(manip_jacob_i*manip_jacob_i' + damping_factor*(Um*Um') );
            
             %dqt1 = pinv(manip_jacob_i)*km*symmat2vec(mdiff_i);
            dqt1 = manip_invmatrix*(km/err)*symmat2vec(mdiff_i);
            %dqt1 = manip_invmatrix*(km)*symmat2vec(mdiff_i);

            joints_i= joints_i + dqt1*dt;
            
            maniplist=[maniplist; 0 reshape(manip_i, 1,9)];
        end
    
        csvwrite("/home/nnrthmr/Desktop/manips2.csv", maniplist);

        % Saving results
        [U,S,V] = svd(manip_i);
        scales(1,i) = min(diag(S));
        positions(i,:) = rhuman.getPos(joints_i);

        % Plotting
        x = 1:size(errs,2) ; 
        for k = 1:size(errs,1)
           plot(x,errs(k,:))
        end
        
        % printing results
        'after'
        manip_desired
        manip_i
        
        S =  eig(manip_i);
        ve= prod(sqrt(S))*(4.0/3.0)*pi
        %manip_desired = eye(3) * nthroot(ve/(4.0/3.0*pi),3)^2;

        manip_desired = eye(3) * nthroot(ve/(4.0/3.0*pi),2)^2; % singularity in one direction
        %manip_desired(1,1)=1e-6;

        Sd =  eig(manip_desired);
        vd= prod(sqrt(Sd))*(4.0/3.0)*pi %vd must be equal to ve
    end
    
    positions
    scales
    
end
