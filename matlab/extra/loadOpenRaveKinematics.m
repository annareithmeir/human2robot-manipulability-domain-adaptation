%% EXAMPLE FOR ICRA-18 WORKSHOP ON ERGONOMICS - Function File
%--------------------------------------------------------------------------
%  This example is part of the manuscript entitled:
% 
%  asdfasdfasdfasdfasdfasdfasdfasdfasdf
% 
%  Submitted to the ICRA-WS 2018 Workshop
%  Date:    January, 2018
% 
%  Authors: Luis F C Figueredo
%           Lipeng Chen
%           Mehmet Dogar
%           
%  Contact information: figueredo@ieee.org
%-------------------------------------------------------------------------- 
% 
% Configure positioning of the arm (constraint it to be close to the robot)
% 
% Function inputs:
%       humanShoulder2World -  pose of the shoulder with regard to the world frame (default: DQ(1) where human faces Y). 
%       length_upperarm  - Length of the upperarm (default:  0.3102)
%       length_forearm   - Length of the forearm (default:  0.2617)
%       length_hand      - Length of the hand (default: 0 OR 0.095; % Assume 9.5cm up to the pressure point in the human hand 
% 
% Function outputs:
%       humankine - Struct from DQ_Kinematics that describes the arm kinematics
%       humangeom - Struct with humangeom configuration (joint limits, etc
% 
%-------------------------------------------------------------------------- 


%% Function icraWS_fctLoadHuman2
% 
function [humankine, humangeom] = icraWS_fctLoadHuman2(humanShoulder2World, length_upperarm, length_forearm, length_hand)



% Size of human limb
humangeom.size.UpperArm  = length_upperarm;  %( Default: 0.3102; )
humangeom.size.ForeArm   = length_forearm;   %( Default: 0.2617; )
humangeom.size.Hand      = length_hand;      %( Default: 0; )
humangeom.size.totalRange= humangeom.size.UpperArm + humangeom.size.ForeArm + humangeom.size.Hand;
    
% Pose of human shoulder (kine base) to the world frame
humangeom.pos.shoulder2world = humanShoulder2World;
    
% Joint torque limits
% humangeom.joint.limits_upper = [140  +80  +80  +140  +26   +20   +26]'*(pi/180);  
% humangeom.joint.limits_lower = [+5   -80  -40  +0    -20   -10   -20]'*(pi/180);
% humangeom.joint.limits_upper = [140  +80  +80  +140  +13   +10   +13]'*(pi/180);  
% humangeom.joint.limits_lower = [+5   -80  -40  +0    -10   -5   -10]'*(pi/180);


% humangeom.joint.limits_upper = [140  +80  +80  +140  +60   +5   +5]'*(pi/180);  
% humangeom.joint.limits_lower = [+5   -80  -40  +0    -40   -5   -5]'*(pi/180);

% humangeom.joint.limits_upper = [140  +80  +80  +140  +60   +10  +25]'*(pi/180);  
% humangeom.joint.limits_lower = [+5   -80  -40  +0    -40   -5   -20]'*(pi/180);


%%%%=[ Optitrack capture ]
humangeom.joint.limits_upper = [140  +80  +80  +140  +60   +40  +80]'*(pi/180);  
humangeom.joint.limits_lower = [+5   -80  -40  +0    -40   -20   -60]'*(pi/180);




% TORQUE PARAMETERS
torqueMatrix = fctBuildMaxTorqueTablePerJoint;
humangeom.joint.jointmaxTorqueTablePerJoint = torqueMatrix;
humangeom.jointmaxtorque     = @fctMaxTorque;

% TO-DO: WRITE
humangeom.joint.jointmaxtorqueTask = [];



%==========================[ HUMAN ARM - DH ]===========================
Lupper = humangeom.size.UpperArm;
Llower = humangeom.size.ForeArm;
Lhand  = humangeom.size.Hand;
DQ_DH.d  =    [0,     0,     -Lupper,       0,        Llower,     0,        0       Lhand];
DQ_DH.theta = [0,    -pi/2,  -pi/2,         0,        0,          -pi/2,    pi/2    0];
DQ_DH.a =     [0,     0,      0,            0,        0,          0,        0       0];
DQ_DH.alpha = [-pi/2,+pi/2,  -pi/2,        -pi/2,    -pi/2,       pi/2,     pi/2    0];
DQ_DH.dummy =    [0,0,0,0,0,0,0,    1];
       

DQMatrix = [DQ_DH.theta;    DQ_DH.d;  ... 
            DQ_DH.a;        DQ_DH.alpha;    DQ_DH.dummy];
humankine = DQ_kinematics(DQMatrix,'standard');


%==========================[ HUMAN ARM - CONTROLLER ]======================
humangeom.control = @controlHuman;
humangeom.controlconfig.plot = false;
humangeom.controlconfig.curJoint = zeros(7,1);

humangeom.controlpos = @controlHumanPos;






%% Control human arm (IK map to desired pose)
function thetaout = controlHumanPos(xdpose, theta)
 
        
%============[    Configuration   
itmax   = 100;
epsilon = 0.005;
KGAIN = 0.25;

it=0;
error = epsilon + 1;
C8 = diag([-1 +1 +1 +1, -1 +1 +1 +1]);
liminf = humangeom.joint.limits_lower;
limup  = humangeom.joint.limits_upper;
thetaout = theta;

% options = optimoptions('fmincon','Display','off','Algorithm','sqp');
options = optimoptions('fmincon','Display','off');

%--------------------
thetamedio = 0.5*(liminf + limup);
%--------------------
% pause


xd = xdpose.translation;

while norm(error) > epsilon
    % Stop criteria 2
    it = it+1;
    if (it > itmax), break, end
    
    xm = humankine.fkm(thetaout);   
    xpos = xm.translation;
    
    % Kinematics Error and Jacobian
    error = vec4(xpos - xd);         
    jacobpos = humankine.jacobp( humankine.jacobian(thetaout), xm);        
    jacob_pseudo = jacobpos'*pinv(jacobpos*jacobpos'+0.001*eye(size(jacobpos,1)));
    thetaout = thetaout - jacob_pseudo*KGAIN*error;

%     thetaOLD = thetaout;
% 
%     GAINX = 0.2;
%     thetaout= lsqlin(jacob, +( GAINX*error - jacob*thetaOLD ),[],[],[],[],liminf,limup,thetaOLD,options);
    
    
% % %     theta2arm = linprog(0.5*jacob,[],[],[],[],liminf,limup);
%     thetaOLD = thetaout;
%     GAINX = 0.5;
%      fun = @(q)(( GAINX*error + jacob*( q - thetaOLD ) )'*( GAINX*error+ jacob*( q - thetaOLD ) ));    
%     x0 = thetaOLD;
%     A=[];b=[];Aeq=[];beq=[];nonlcon=[];
%     thetaout = fmincon(fun,x0,A,b,Aeq,beq,liminf,limup,nonlcon,options);
% %     SIM.theta =    thetaout;     
%     [it norm(error)]
    jacob = jacobpos;
    nerror = 0;
    for i=1:7, nerror = nerror + .5*(thetaout(i)-thetamedio(i))^2;  end;    

    nproj  = eye(size(jacob,2))-jacob_pseudo*jacob;
    njacob = (thetaout' - (thetamedio)');
      
    nPseudoInv = (njacob*nproj);
    nPseudoInv = nPseudoInv'*pinv( nPseudoInv*nPseudoInv'  +  0.01*eye(size(nPseudoInv,1)));        
    
    thetaout = thetaout + 0.1*nPseudoInv*(-0.1*nerror -njacob*jacob_pseudo*KGAIN*error );
    
%     thetaout = thetaout + 0.2*nPseudoInv*(-0.1*nerror );
% size( 0.1.*nproj*(-0.1*nerror ) )

%     theta2arm = theta2arm + 0.1.*nproj*(njacob*nerror );
    
    % Plot the arms
%     if humangeom.controlconfig.plot
        plot(humankine,thetaout);
%         plot(xm)
%         plot(xdpose);
%         pause(0.1)
        %plot small coordinate systems such that one does not mistake with the desired absolute pose,
        %which is the big frame
%         plot(baxterkine.twoarms,theta2arm,'scale',0.1); 
%     end    
   
    
end
% disp('teste')
[norm(error) it]
% pause

end 



%% Control human arm (IK map to desired pose)
function thetaout = controlHuman(xd, theta)
 
        
%============[    Configuration   
itmax   = 100;
epsilon = 0.005;
KGAIN = 0.25;

it=0;
error = epsilon + 1;
C8 = diag([-1 +1 +1 +1, -1 +1 +1 +1]);
liminf = humangeom.joint.limits_lower;
limup  = humangeom.joint.limits_upper;
thetaout = theta;

% options = optimoptions('fmincon','Display','off','Algorithm','sqp');
options = optimoptions('fmincon','Display','off');

%--------------------
thetamedio = 0.5*(liminf + limup);
%--------------------
pause

while norm(error) > epsilon
    % Stop criteria 2
    it = it+1;
    if (it > itmax), break, end
    
    xm = humankine.fkm(thetaout);     
    
    % Kinematics Error and Jacobian
    error = vec8(DQ(1) - xm'*xd);         
    jacob = haminus8(xd)*C8*humankine.jacobian(thetaout);        
    jacob_pseudo = jacob'*pinv(jacob*jacob'+0.001*eye(size(jacob,1)));
    thetaout = thetaout - jacob_pseudo*KGAIN*error;

%     thetaOLD = thetaout;
% 
%     GAINX = 0.2;
%     thetaout= lsqlin(jacob, +( GAINX*error - jacob*thetaOLD ),[],[],[],[],liminf,limup,thetaOLD,options);
    
    
% % %     theta2arm = linprog(0.5*jacob,[],[],[],[],liminf,limup);
%     thetaOLD = thetaout;
%     GAINX = 0.5;
%      fun = @(q)(( GAINX*error + jacob*( q - thetaOLD ) )'*( GAINX*error+ jacob*( q - thetaOLD ) ));    
%     x0 = thetaOLD;
%     A=[];b=[];Aeq=[];beq=[];nonlcon=[];
%     thetaout = fmincon(fun,x0,A,b,Aeq,beq,liminf,limup,nonlcon,options);
% %     SIM.theta =    thetaout;     
%     [it norm(error)]
    
    nerror = 0;
    for i=1:7, nerror = nerror + .5*(thetaout(i)-thetamedio(i))^2;  end;    

    nproj  = eye(size(jacob,2))-jacob_pseudo*jacob;
    njacob = (thetaout' - (thetamedio)');
      
    nPseudoInv = (njacob*nproj);
    nPseudoInv = nPseudoInv'*pinv( nPseudoInv*nPseudoInv'  +  0.01*eye(size(nPseudoInv,1)));        
    
    thetaout = thetaout + 0.1*nPseudoInv*(-0.1*nerror -njacob*jacob_pseudo*KGAIN*error );
    
%     thetaout = thetaout + 0.2*nPseudoInv*(-0.1*nerror );
% size( 0.1.*nproj*(-0.1*nerror ) )

%     theta2arm = theta2arm + 0.1.*nproj*(njacob*nerror );
    
    % Plot the arms
%     if humangeom.controlconfig.plot
        plot(humankine,thetaout);
%         plot(xm)
        plot(xd);
%         pause(0.1)
        %plot small coordinate systems such that one does not mistake with the desired absolute pose,
        %which is the big frame
%         plot(baxterkine.twoarms,theta2arm,'scale',0.1); 
%     end    
   
    
end
% disp('teste')
[norm(error) it]
% pause

end 



%% get TORQUE MATRIX (given joint)  : MEAN VALUE
function maxTorqueMatrix = fctMaxTorque(theta)

    torque = zeros(7,1);

    for it = 1:7
        indexMin  =  max(1, sum( theta(it)*180/pi > torqueMatrix(it).joints ));
        indexMax  =  min(indexMin+1, torqueMatrix(it).length);

        torque(it) = 0.5*(torqueMatrix(it).tmean(indexMin)  +  torqueMatrix(it).tmean(indexMax) );
    end

    % torque = torque*inv(norm(torque));
    maxTorqueMatrix = diag(torque);

end 


end


%% static function - Returns Torque Matrix

function torqueStructMatrix = fctBuildMaxTorqueTablePerJoint()
%   

% Table of data
tau(1).joints = [5      10    20    40    60    80   100    120   140];
tau(1).tPlus  = [20.89 15.09 17.98 17.80 18.54 18.19 17.00 14.36 12.56];
tau(1).tMinus = [19.56 17.89 17.42 22.72 27.89 29.64 27.47 33.26 32.77];
tau(1).tmean  = 0.5*(tau(1).tPlus + tau(1).tMinus);
tau(1).length = 9;

tau(2).joints = [-80   -60   -40   -20   0     20    40    60    80];
tau(2).tPlus  = [13.56 13.68 15.90 18.05 28.01 24.76 21.90 18.43 18.33];
tau(2).tMinus = [ 6.42  5.83  6.58  9.82 17.45 19.98 18.18 18.79 21.28];
tau(2).tmean  = 0.5*(tau(2).tPlus + tau(2).tMinus);
tau(2).length = 9;

tau(3).joints = [-40   -20   0     20    40    60    80];
tau(3).tPlus  = [12.37 15.05 17.20 17.42 15.12 14.05 15.50];
tau(3).tMinus = [14.03 15.45 16.35 18.49 16.07 13.78 13.02];
tau(3).tmean  = 0.5*(tau(3).tPlus + tau(3).tMinus);
tau(3).length = 7;

tau(4).joints = [0     20    40    60    80    100   120   140];
tau(4).tPlus  = [12.10 12.85 14.39 17.27 20.54 17.93 17.23 10.60];
tau(4).tMinus = [ 6.23  8.80 11.83 14.11 15.57 13.82 12.34 11.82];
tau(4).tmean  = 0.5*(tau(4).tPlus + tau(4).tMinus);
tau(4).length = 8;

tau(5).joints = [-60   -40   -20   0     20    40    60    80];
tau(5).tPlus  = [0.33  0.91  1.36  1.80  1.97  2.08  2.33  2.27  ];
tau(5).tMinus = [3.62  3.40  3.63  3.50  3.00  1.32  0.87  0.29  ];
tau(5).tmean  = 0.5*(tau(5).tPlus + tau(5).tMinus);
tau(5).length = 8;

tau(6).joints = [-20    0    20    40];
tau(6).tPlus  = [7.22  6.70  5.44  4.28];
tau(6).tMinus = [3.47  4.18  5.49  6.42];
tau(6).tmean  = 0.5*(tau(6).tPlus + tau(6).tMinus);
tau(6).length = 4;

tau(7).joints = [-60   -40   -20   0     20    40    60    80];
tau(7).tPlus  = [1.97  2.25  3.13  4.14  4.23  4.01  3.39  2.42];
tau(7).tMinus = [1.28  2.16  3.74  3.43  3.92  3.68  3.17  2.06];
tau(7).tmean  = 0.5*(tau(7).tPlus + tau(7).tMinus);
tau(7).length = 8;

torqueStructMatrix = tau;

end


%% NESTED FUNCTION 
% function maxTorqueMatrix = maxtorqueFct(q)
% % POS-CONSTRAINT Summary of this function goes here
% 
%     % Robot base arms position with regard to the human
%     robo_pos = [0; pos2robot.robot2shoulder];
%     dist_max = pos2robot.mindist;
% 
% 
%     % C   = [( jacobpos*( q - theta ) + xpos -robo_pos )'*( jacobpos*( q - theta ) + xpos -robo_pos ) - dist_max ; ...
%     %        ( jacobpos*( q - theta ) + xpos )'*( jacobpos*( q - theta ) + xpos ) - (RANGE_ARM^2)*0.95;  
%     %       ]; 
% 
% 
%             % C =   ( human_pos - robot_pos )^2 <= dist_max  (ball)  
%             % In other words,        
%             % Basically:   norm( predicted_position - robot_pos ) <= dist_max^2
%     C   = [( jacobpos*( q - theta ) + xpos -robo_pos )'*( jacobpos*( q - theta ) + xpos -robo_pos ) - dist_max^2 ;      ]; 
%     Ceq = [];    
% 
% end




