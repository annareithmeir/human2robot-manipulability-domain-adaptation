%% return orientation (quaternion [a;b;c;d]= a + bi + cj + dk) from forward kinematics
% 
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
% Compute orientation forward kinematics from joints 
% 
%-------------------------------------------------------------------------- 
% 
% Function inputs:
%       joints      -   double(7,1) 
% 
% Function outputs:
%       orientation -  double(4,1) with orientation in quaternion [a;b;c;d]= a + bi + cj + dk
%-------------------------------------------------------------------------- 
% 
%  % Computing FKM  (return quaternion [a;b;c;d]= a + bi + cj + dk)
%  rot = getOrientation(joints)
% 
%-------------------------------------------------------------------------- 



%% Get Orientation from Forward Kinematics
function rotationout = getOrientation(obj, theta7)
    rotationout = vec4(obj.kine.fkm(obj.getJoints8(theta7)).P); 
end   

