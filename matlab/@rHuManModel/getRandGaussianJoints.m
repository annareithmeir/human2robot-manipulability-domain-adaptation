%% Returns a 7-joint vector with random (Gaussian) values (within joint limits).
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
% Get random joints (normal distribution with mean in joint-mean (between limits) and std (from joint-range) that satsifies the joint limits. 
%-------------------------------------------------------------------------- 
% Function outputs:

%       joints      -   double(7,1)  
%       ['seed',seedopt] - Defines the seed for the random generator (e.g., getRandGaussianJoints('seed','shuffle')) [default: shuffle]
%                          After use it returns to previous rng.   
%-------------------------------------------------------------------------- 
% 
%  joints = getRandGaussianJoints()
%         = getRandGaussianJoints('seed','default'); => Initializes Mersenne Twister generator with seed 0. This is the default setting at the start of each MATLAB session.
% 
%-------------------------------------------------------------------------- 



%% Get random joint configuration
% Get random (Gaussian) joints (within limits).
function jointsout = getRandGaussianJoints(obj, options)
    arguments
        obj
        options.seed;         
    end
    
    rngOld = rng; 
    if isfield(options,'seed')
        rng(options.seed);
        obj.printv(obj,'Setting up rng seed');
    else
        rng('shuffle');        
    end
    
    % get random joints (mean in joint means 
    jointsout = 0.5*(randn(7,1)).*obj.kineconfig.joint_range + obj.kineconfig.joint_mean;
    % Not allow to go over maximum
    jointsout = min(jointsout, obj.kineconfig.joint_upperlimits);
    % Not allow to go over minimum
    jointsout = max(jointsout , obj.kineconfig.joint_lowerlimits);
    
    rng(rngOld);    
    
    %jointsout = min(max((randn(7,1)+1).*obj.kineconfig.joint_mean,obj.kineconfig.joint_upperlimits) , obj.kineconfig.joint_lowerlimits);
    %obj.kineconfig.joint_lowerlimits
end   
