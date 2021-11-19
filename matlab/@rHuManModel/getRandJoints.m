%% Returns a 7-joint vector with uniformally random values (within joint limits).
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
% Get random joints (uniform distribution within joint limits). 
%-------------------------------------------------------------------------- 
% Function outputs:
% 
%       joints      -   double(7,1)  
%       [Optional:logical(7,1)]   - where 0 implies joint=mean(between limits) and 1 implies rand values.
%       [Optional:'seed',seedopt] - Defines the seed for the random generator (e.g., getRandGaussianJoints('seed','shuffle')) [default: shuffle] 
%                          After use it returns to previous rng.
%       [Optional:'length',N]     - Using 'length' followed by N where N is the length for output random values, i.e., randjoints double(7,N)
%-------------------------------------------------------------------------- 
% 
%  joints = getRandJoints(joints)
%         = getRandJoints(joints, [0; 0; 0; 0;  1; 1; 1])  %only wrist random with joint-mean(between limits) in shoulder and elbow-flexion joints
%         = getRandJoints('seed','default'); => Initializes Mersenne Twister generator with seed 0. This is the default setting at the start of each MATLAB session.
%         = getRandJoints('length',1000);    => Returns 1000 random joints double(7,1000).  
%         = getRandJoints([0; 0; 0; 0;  1; 1; 1],'length',1000);    => Returns 1000 random joints double(7,1000) where only the last 3-joints are random..
%         
%-------------------------------------------------------------------------- 


%% Get random joint configuration
% Get random (uniform) joints (within limits). 
% Optional logical (7,1) where 0 implies joint=0 and 1 implies rand. 
function jointsout = getRandJoints(obj,logtheta7,options)
    arguments
        obj
        logtheta7  = logical(zeros(7,1));
        options.seed;
        options.length double = 1;
    end
    
    rngOld = rng; 
    if isfield(options,'seed')
        rng(options.seed);
        obj.printv(obj,'Setting up rng seed');
    else
        rng('shuffle');        
    end
    
    jointsout = zeros(7,options.length);
    %if exist('logtheta7','var') 
    if norm(double(logtheta7)) > 0
        % Getting mean position for non-random joints
        jointsout(~logical(logtheta7),:) = obj.kineconfig.joint_mean(~logical(logtheta7)).*ones(1,options.length);
        
        % Getting random positions and add with previous joints.
        jointsout(logical(logtheta7),:)  = rand(sum(double(logtheta7)),options.length).*obj.kineconfig.joint_range(logical(logtheta7)) + obj.kineconfig.joint_lowerlimits( logical(logtheta7) );
        
    else
        
        jointsout = rand(7,options.length).*obj.kineconfig.joint_range + obj.kineconfig.joint_lowerlimits;
    end
    
    rng(rngOld);        
end
