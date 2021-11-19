%% OpenSim: Kinematics
%--------------------------------------------------------------------------
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
% 
% Create human kinematics for the model used in OpenSim (Saul et al, 2015)
% 
% Function inputs:
%       shoulderPose     -  pose of the shoulder with regard to the world frame (default: DQ(1) where human faces Y). 
%       length_upperarm  - Length of the upperarm (default:  0.3102)
%       length_forearm   - Length of the forearm  (default:  0.2617)
%       length_hand      - Length of the hand  
% 
% Function outputs:
%       humankine - Struct from DQ_Kinematics that describes the arm kinematics
%       humangeom - Struct with humangeom configuration (joint limits, etc
%       obj.osimmodel     - Coupled of functions from the specificities from the opensim model  
%          obj.osimmodel.fkm     : FKM (7-joints [q] => 8-joints [q'] humankine FKM) 
%                          Differences: [ q'(3)  <=  +q(1)    ]
%                                       [ q'(4)  <=  -q(3)    ]
%                                       [ q'(5:8) <=  +q(4:7) ]
%          obj.osimmodel.jacobian: Jacobian (7-joints to 8-joints humankine as described above) 
%                          Same differences, but Jout[1] = Jin[1]+J[3]
% 
%-------------------------------------------------------------------------- 
%   ====( CHANGES LOG )====         
%   [ 04/07/2019 ] - Updated the dumb joint at the end to be a
%   transformation making the hand to have the x-axis (hand) facing palm-in
%   and the z-axis (hand) facing from the pulse to the fingers. 
%-------------------------------------------------------------------------- 




%% Function load (outputs) kinematics 
% 
% function [humankine, humangeom, opSim] = humanOpenSimKinematics(shoulderPose, length_upperarm, length_forearm, length_hand, joint_limits)
function build_humanOpenSimKinematics(obj, shoulderPose, length_upperarm, length_forearm, length_hand, joint_limits)


% Size of human limb
obj.kineconfig.UpperArm  = length_upperarm;  %( Default: 0.3102; )
obj.kineconfig.ForeArm   = length_forearm;   %( Default: 0.2617; )
obj.kineconfig.Hand      = length_hand;      %( Default: 0; )
obj.kineconfig.totalRange= obj.kineconfig.UpperArm + obj.kineconfig.ForeArm + obj.kineconfig.Hand;
    
% Pose of human shoulder (kine base) to the world frame
obj.kineconfig.shoulderPose = shoulderPose;
    

% JOINT LIMITS
obj.kineconfig.joint_lowerlimits = joint_limits(:,1); 
obj.kineconfig.joint_upperlimits = joint_limits(:,2); 
obj.kineconfig.joint_mean  = (obj.kineconfig.joint_lowerlimits + obj.kineconfig.joint_upperlimits)*0.5;
obj.kineconfig.joint_range = (obj.kineconfig.joint_upperlimits - obj.kineconfig.joint_lowerlimits);


%                 options.jointlimits        (7,2) double =  [[-90  0    -80   0        -90     0  -60];[130  180   +40  130      +90   +25  +60]]'*(pi/180);                
%                 options.joint_lowerlimits  (7,2) double =   [-90  0    -80   0        -90     0  -60]'*(pi/180);                
%                 options.joint_upperlimits  (7,2) double =   [130  180   +40  130      +90   +25  +60]'*(pi/180);
obj.kineconfig.checkJointLim = @fctCheckJointLimits;
obj.kineconfig.joints7to8 = @(theta7)( [theta7(1);  theta7(2);  theta7(1);  -theta7(3);    theta7(4); theta7(5); theta7(6); theta7(7);  ] ) ;
obj.kineconfig.joints8to7 = @(theta8)( [theta8(1);  theta8(2);  -theta8(4);   theta8(5); theta8(6); theta8(7); theta8(8);   ] ) ;



%% OpenSim Model
% IMPORTANT: Note that THIS IS A 8-JOINT SYSTEM where joint 4 is equal to joint 3 must be always substracted from joint 1.
%            joint 3 is equal to joint 1
%            joint 4 to 8 would be equal to joints 3-7  
%                   (but new joint 4 must be set to negative)
% 


%==========================[ HUMAN ARM - DH ]===========================
% Parameters
Lupper = obj.kineconfig.UpperArm;
Llower = obj.kineconfig.ForeArm;
Lhand  = obj.kineconfig.Hand;


% 
% D-H Creation
DQ_DH.d  =    [0,      0,      0,  Lupper,       0,    -Llower,        0,    0          -Lhand];
DQ_DH.theta = [0,      0,      0,   +pi/2,        0,        0,          +pi/2,    pi/2     pi];
DQ_DH.a =     [0,      0,      0,   0,            0,        0,          0,        0         0];
DQ_DH.alpha = [pi/2,   +pi/2,   0,  +pi/2,        +pi/2,    +pi/2,      -pi/2,    pi/2     pi];
DQ_DH.dummy =    [0,0,0,0,0,0,0,0,    1];

DQMatrix = [DQ_DH.theta;    DQ_DH.d;  ... 
            DQ_DH.a;        DQ_DH.alpha;    DQ_DH.dummy];
obj.kine = DQ_kinematics(DQMatrix,'standard');
obj.kine.base = DQ(1);
obj.kine.base = obj.kineconfig.shoulderPose;


%% Additional functions for opensim
% spfkm and spjacobian to be deprecated 
%     See the correct functions below 
obj.osimmodel.spfkm = @(theta) obj.kine.fkm( theta + [0;0;theta(1);-2*theta(3); zeros(4,1)] ) ;
obj.osimmodel.spjacobian = @jacobianOsim8;



% FKM correct function 
% -------------------------
%    Takes the 7-joints as in OpenSim and outputs the correct FKM 
%    Note the classic D-H parameter model requries 8 joint with the joint 4
%    being equal to joint 1 (removing the rotation performed in joint 1).
%    Also there is an error in the (input) joint 3 which should have been
%    negative (error in the 8-joint FKM I created and need to be fixed). As
%    a hack, I am setting it manually to negative in the fkm function below
% obj.osimmodel.fkm = @(theta) obj.kine.fkm( theta + [0;0;theta(1);-2*theta(3); zeros(4,1)] ) ;
obj.osimmodel.fkm = @FKM_opensim;
obj.osimmodel.jacobian = @jacobianOsim7;
obj.osimmodel.jacobgeom = @GeometricJacobianOsim;



% obj.kine,  






%% KINEMATIC FUNCTIONS
% FKM correct function 
% -------------------------
function xm = FKM_opensim(theta7, argextra)    
    theta8 = [theta7(1);  theta7(2);  theta7(1);  -theta7(3);    theta7(4); theta7(5); theta7(6); theta7(7);  ];
    extrainput = false;
    if exist('argextra','var')
        if isnumeric(argextra)
            extrainput = true;
        end
    end
    if extrainput
        xm = obj.kine.fkm( theta8, argextra ) ;
    else
        xm = obj.kine.fkm( theta8 ) ;
    end
end


% Geometric Jacobian
% -------------------------
function JacobGeomOut = GeometricJacobianOsim(theta7)
    % CONSTANT MATRICES
    C8 = diag([-1 ones(1,3) -1 ones(1,3)]');
    C4m = -C8(1:4,1:4);    
    
    % Pose and Analytical Jacobian
    xm = FKM_opensim(theta7);
    jacob = jacobianOsim7(theta7); 
    
    % Transformation
    G1 = [2*haminus4( xm.P' )        zeros(4,4)];
    G2 = [2*C4m*haminus4( xm.D' )    2*haminus4( xm.P' )];
    jacobGeom = [G1; G2]*jacob;
    
    % Output Geometric Jacobian 6 x n
    JacobGeomOut = jacobGeom([2:4,6:8],:);
 
end


% Jacobian 7-JOINT
% -------------------------
function JacobianOut = jacobianOsim7(theta7)    
    theta8 = [theta7(1);  theta7(2);  theta7(1);  -theta7(3);    theta7(4:7)  ];
    jacobTmp = obj.kine.jacobian( theta8  );
    %JacobianOut = [(jacobTmp(:,1)) jacobTmp(:,2) jacobTmp(:,4:8) ];
    JacobianOut = [(jacobTmp(:,1)+jacobTmp(:,3)) jacobTmp(:,2) -jacobTmp(:,4)  jacobTmp(:,5:8) ];
end


% Jacobian 8-JOINT
% -------------------------
function JacobianOut = jacobianOsim8(theta8)
    % Expects the correct theta8  (where theta8(3)=theta8(1) and 
    %                                    theta8(4) has inverted sign)
    jacobTmp = obj.kine.jacobian( theta8  );
    %JacobianOut = [(jacobTmp(:,1)+jacobTmp(:,3)) jacobTmp(:,2) jacobTmp(:,4:8) ];
    JacobianOut = [(jacobTmp(:,1)+jacobTmp(:,3)) jacobTmp(:,2) -jacobTmp(:,4)  jacobTmp(:,5:8) ];
%     JacobianOut(:,1) = JacobianOut(:,1) + vec8( xmTmp12*( DQ( jacobTmp(:,1) )*xmTmp )*xmTmp12'*xmTmp' );    

end






%% Check Joint Limits
function [boolJointLimit, lowerLimit, upperLimit] = fctCheckJointLimits(theta_cur,offset)
    
    lowerlim = obj.kineconfig.joint_lowerlimits;
    upperlim = obj.kineconfig.joint_upperlimits;
    if exist('offset','var') 
        lowerlim = lowerlim -offset*abs(obj.kineconfig.joint_lowerlimits);
        upperlim = upperlim +offset*abs(obj.kineconfig.joint_upperlimits);
    end
    lowerLimit = theta_cur < lowerlim;
    upperLimit = theta_cur > upperlim;
    boolJointLimit = sum(double(lowerLimit)+  double(upperLimit))>0;
end



end

