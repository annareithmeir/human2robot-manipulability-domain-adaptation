%% return pose in dual quaternion (forward kinematics)
% 
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
% Compute forward kinematics from joints 
% 
%-------------------------------------------------------------------------- 
% Function inputs:
%       joints      -   double(7,1) 
% 
% Function outputs:
%       pose     -  class DQ with orientation and position (in unit dual quaternions) 
%-------------------------------------------------------------------------- 
% 
%  % Computing FKM  (return class DQ)  
%  pose = getFKM(joints)
% 
%-------------------------------------------------------------------------- 



%% Get Forward Kinematics
% 
function poseout = getFKM(obj, theta7) 
    poseout = obj.kine.fkm(obj.getJoints8(theta7)); 
end

