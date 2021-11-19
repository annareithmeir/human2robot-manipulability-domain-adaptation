%% return position [x;y;z] from forward kinematics
% 
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
% Compute position forward kinematics from joints 
% 
%-------------------------------------------------------------------------- 
% Function inputs:
%       joints      -   double(7,1) 
% 
% Function outputs:
%       position    -  double(3,1) with position
%-------------------------------------------------------------------------- 
% 
%  % Computing FKM  (return [x;y;z])  
%  position = getPos(joints)
% 
%-------------------------------------------------------------------------- 



%% Get Position from Forward Kinematics
% 
function positionout = getPos(obj, theta7)
    positionout = obj.kine.fkm(obj.getJoints8(theta7)).translation.q(2:4); 
end

