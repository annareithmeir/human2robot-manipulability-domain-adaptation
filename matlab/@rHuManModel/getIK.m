%% return joint configuration (inverse kinematics)
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
% Compute inverse kinematics from pose (or position)
%-------------------------------------------------------------------------- 
% Function inputs:
%         pose     -  DQ (class DQ included in this package - from DQrobotics package) 
%     OR  pose     -  Double (7,1) - for [quaternion;position] = [[a;b;c;d];  x;y;z]
%     OR  orientation -  Double (4,1) - for quaternion only [a;b;c;d]= a + bi + cj + dk
%     OR  position    -  Double (3,1) - [x;y;z] (for position only)
% 
% Function outputs:
%       joints - double(7,1) of joints
%       output - optimization output (multistart sqp)
%-------------------------------------------------------------------------- 
% 
% %  Example (orientation and position)
%       
%       % Building pose (Example)
%       phi=pi/2;
%       position = [0.25; 0; 0.25];
% 
%       % Defining rotation quaternion 
%       pose = DQ([cos(phi) sin(phi) 0 0]);  
% 
%       % Defining position quaternion 
%       pose = pose + DQ.E*(0.5)*DQ([0;position]);  
% 
%  % Finding IK
%  q = getIK(pose)
%  [q, output] = getIK(pose)
% 
% %  Example (only position (any orientation))
% 
%  % Finding IK 
%  position = [0.25; 0; 0.25]; 
%  q = getIK([0; position])
%  [q, output] = getIK([0; position])
% 
%-------------------------------------------------------------------------- 



%% Function load (outputs) kinematics 
% 
function [jointsout, output] = getIK(obj, pose)
    
    if ~isa(pose,'DQ')
        obj.mustBeA(pose,'double'); 
        
        full_pose = 0 ; 
        
        % Position only
        if size(pose,1)==3     && size(pose,2)==1
            pose = DQ(1) + DQ.E*0.5*DQ([0; pose]);
        elseif size(pose,2)==3 && size(pose,1)==1
            pose = DQ(1) + DQ.E*0.5*DQ([0; pose']);
            
        % Rotation only    
        elseif length(pose)==4  
            if abs(1-norm(pose))>10^-6  
                obj.printv(obj,['*** Warning: Input quaternion not normalized. Normalization has been done internally.'])                
                pose = pose*inv(norm(pose));
            end            
        % Full pose (in vec)
        elseif size(pose,1)==7     && size(pose,2)==1
            full_pose = 1 ; 
            rot4 = DQ(pose(1:4));
            pos4 = DQ([0; pose(5:7)]);            
        elseif size(pose,1)==1     && size(pose,2)==7
            full_pose = 1 ; 
            rot4 = DQ(pose(1:4));
            pos4 = DQ([0; pose(5:7)']);
            
        % Full pose (in vec)
        elseif size(pose,1)==8     && size(pose,2)==1
            full_pose = 1 ; 
            rot4 = DQ(pose(1:4));
            pos4 = DQ(pose(5:8));
        elseif size(pose,1)==1     && size(pose,2)==8
            full_pose = 1 ; 
            rot4 = DQ(pose(1:4));
            pos4 = DQ(pose(5:8));            
        else
            error([10,9,'*** getIK(P) where P=pose (in DQ) OR P=double(3,1) for position OR P=double(4,1) for orientation (quaternion: [a;b;c;d]= a + bi + cj + dk])  OR P=double(7,1) for pose [orientation; position]'])
        end
        % Computing full pose
        if full_pose == 1
            if abs(1-norm(vec4(rot4)))>10^-6  
                obj.printv(obj,['*** Warning: Input quaternion not normalized. Normalization has been done internally.'])                
                rot4 = rot4*inv(norm(rot4));
            end  
            pose = rot4 + DQ.E*0.5*(pos4)*rot4;
        end
    end    
    jointsout = zeros(7,1);
    ms = MultiStart('Display','off');
    fun = @(theta)( norm(vec4(DQ(1) - pose*obj.getFKM(theta)') )  ); 
    for i=1:5
        opts = optimoptions(@fmincon,'Algorithm','sqp','MaxFunEvals',(500*i),'ConstraintTolerance',10^-5,'OptimalityTolerance',10^-7,'display','off','MaxIter',100*i);                            
        problem = createOptimProblem('fmincon','objective',fun,'x0',obj.kineconfig.joint_mean+0.1*randn(7,1),'lb',obj.kineconfig.joint_lowerlimits,'ub',obj.kineconfig.joint_upperlimits,'options',opts);                      
        [res.theta, res.fval,res.exitflag,res.output] = run(ms,problem,3+2*i);     
        if res.exitflag>0 && res.fval<0.01
            jointsout = res.theta;
            break                    
        end
    end
    output = res;

end

