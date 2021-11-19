%% Function that maps the joints from OpenRave to OpenSim
%--------------------------------------------------------------------------
% 
%  It takes the model from the 7 DOF arm from OPenRave to the 7 DOF arm
%  from OpenSim. 
% 
%  Details: There are quite a lot of differences, perhaps the most
%  remarkable one is the first joint in Opensim where the axis points to
%  the global z-axis (similarly to joint 3). To allow 3-DOF in the shoulder
%  to be independent, it requires the joint3 to be q3 = q3desired + q1
% 
%-------------------------------------------------------------------------- 
% 
% Function inputs:
%       raveModel   - OpenRave model created w/ fctLoadHuman_openRAVE (dqKinematics object)
%       theta_input - Joint config. from OpenRave model (vector 7x1)
% Function outputs:
%       theta_output  - Joint config. from OPENSIM model% 
%-------------------------------------------------------------------------- 
% Date: 4/12 - Luis
%-------------------------------------------------------------------------- 

function theta_output = fctModel_Rave2Sim_a(raveModel, theta_input)
%   Getting the shoulder pose (After 2DOF) from the RAVE-MODEL
    x12 = raveModel.fkm(theta_input , 2);
%     x12 = armteste.fkm( -double([[1:7]==1])'*pi/2  -double([[1:7]==2])'*pi/4  + double([[1:7]==5])'.*pi     , 2)

%   Getting the high of the arm (combined rotation around x and y) given by
%   the angle in the z axis (final to base)
%     a2 =  pi - acos( vec4( x12'*DQ.k*x12 )'*vec4(DQ.k)  );    
    a2 =  acos( vec4( x12'*DQ.k*x12 )'*vec4(DQ.k)  );    
    if theta_input(2)>pi/2
        a2 = -a2; 
    end
%   Doing the same for the x-axis (defines the 1st joint in oSIM model)    
%     a1 = pi/2 - acos( vec4( x12'*DQ.i*x12 )'*vec4(DQ.i)  ); 
x12 = raveModel.fkm(theta_input , 2);
    a1 = - acos( vec4( x12'*DQ.k*x12 )'*vec4(DQ.i)  ); 
    
    
    
    
    postemp = raveModel.fkm(theta_input , 3).translation -raveModel.base.translation;
    a1 = atan( postemp.q(3)/postemp.q(2) );
    if theta_input(1)>pi/2
        a1 = pi + a1;
    end
%     acos( vec4( x12'*DQ.k*x12 )'*vec4(DQ.i)  )*180/pi    
    
%   Doing the same for the last axis (y-axis)
%     a3 = pi -acos( vec4( x12'*DQ.j*x12 )'*vec4(DQ.j)  );
%     acos( vec4( x12'*DQ.j*x12 )'*vec4(DQ.j)  )*180/pi
    
    theta_output = double([[1:7]==1])'.*(a1); 
    theta_output(2) = a2;
    theta_output(3) = theta_input(3) + a1;
%     theta_output(3) = a3
    theta_output(4) = theta_input(4);
%     theta_output(5) = theta_input(5) - pi;
    theta_output(5) = -theta_input(5);
    theta_output(6) = theta_input(6);
    theta_output(7) = theta_input(7);
    
end

