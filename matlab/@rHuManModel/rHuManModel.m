%% RapidHuman-Manipulability Assessment (RHuMAn) - ClassDef rHuManModel.m
%
% CLASS Definition for rHuManModel
% ==========================================================================
% VERSION: VS. 0.3.0  =>  2020-06
% AUTHORS:  Luis F.C. Figueredo, Mehmet Dogar, Anthony Cohn 
%           University of Leeds
% Contact information: figueredo@ieee.org
% ==========================================================================% 
% 
%  RETURNS OBJECT RapidHuman-Manipulability Assessment (RHuMAn)
% 
%  -------------------------------------------------------------
%  Kinematics according to Saul et al., "Benchmarking of dynamic simulation predictions in two software platforms using an upper limb musculoskeletal model Benchmarkin…", CMBBE, (2015).   
%  *** Note that z:height, x:sideways (right shoulder out), y:face-front 
%  ***           for the hand:  z:out of fingers, x:palm down, y:thumbs up
%  - 
% 
%% ====( Output and Inputs )=============================================================
% ------------------------[ OUTPUT ]
% 
%  * RHuMAn object with variables and methods
% 
% ------------------------[ INPUTs (OPTIONAL!) ]
% 
% [Format: String followed by values ]
% ------------[ Kinematics ]
% 
%% Bulleted List
%
%  * 'dq_shoulderBase',double(8,1):     (DQ-pose) indicating the pose of the shoulder wrt the world in unit dual-quaternions [default: [1;zeros(7,1)]] 
%  * 'shoulderHeight',double:           (height)  indicating the position of shoulder wrt the world (in the z-axis). It will be overturned by the option shoulderBase [default: 0]  
%  * 'upperArm',double:                 (length) of the upper-arm limb. [default: 0.302] 
%  * 'foreArm',double:                  (length) of the fore-arm limb.  [default: 0.2795] 
%  * 'hand',double:                     (length) to the center of the hand (where forces will be applied/exerted).  [default: 0.05]                 
%  * 'jointlimits',double(7,2):             (joints) block matrix formed by [joints_min  joints_max]    
%  * 'joint_lowerlimits',double(7,1):       (joints) vector of minimum joint limits. It will be overturned by the option jointlimits [default: [-90  0    -80   0        -90     0  -60]'*(pi/180)]   
%  * 'joint_upperlimits',double(7,1):       (joints) vector of maximum joint limits. It will be overturned by the option jointlimits [default: [130  180   +40  130      +90   +25  +60]'*(pi/180)]
% ------------[ Human collision assessment ]
%  * 'humanbox',double(3,2):         (range)    block matrix building a 3D box for the body [xmin xmax; ymin ymax; zmin zmax] - [default [-0.3145 -0.0355; -0.065 0.065;  -0.60 0.025]] 
%  * 'humanheadCenter',double(3,1):  (position) vector [x;y;z] with center position for the head [default: [-0.20; 0; 0.15]] 
%  * 'humanheadRadius',double:       (length)   radius for the head (depicted as a 3D ball) [default: 0.14] 
% ------------[ Extra ]
%  * 'verbose',logical:  Print all points and steps
% 
%------------------------[ EXAMPLES ]
% newhumanmodel = rHuManModel(); 
% newhumanmodel = rHuManModel('verbose',true);
% newhumanmodel = rHuManModel('dq_shoulderBase',[1 zeros(1,7)]);
% newhumanmodel = rHuManModel('verbose',true,'upperArm',0.35);
% newhumanmodel = rHuManModel('verbose',true,'upperArm',0.35,'foreArm',0.30);
% newhumanmodel = rHuManModel('hand',0);
% newhumanmodel = rHuManModel('shoulderHeight',1.35,'humanheadCenter',[-0.2 0 1.50],'humanbox',[-0.35 0; -0.05 0.05;  0.75 1.3750])
% newhumanmodel = rHuManModel('humanbox',[-0.3145 -0.0355; -0.065 0.065;  -0.60 0.025],'humanheadRadius',0.15);
% newhumanmodel = rHuManModel('joint_lowerlimits',zeros(7,1),'joint_upperlimits',ones(7,1));
%
% 
%% ====( Variables )=============================================================
%     kine          = DQ_kinematics representation for the arm (need external package) 
%     kineconfig    = struct with configurations for the object 
%     hand2tool     = additional transformation from hand to tool (in unit dual quaternions w DQ_kinematics package)  
%     pointsInArm   = array of cell with positions along the arm (for collision detection) 
% 
%% ====( Methods )=============================================================
% 
%     getFKM
%              poseout = getFKM(theta)   => returns the Forward Kinematics for a given joint config theta
% 
%     getIK 
%              jointout = getIK(pose);           => Returns the Inverse Kinematics for a given pose (in unit dual quaternions w DQ_kinematics package)    
%                         getIK(pose);           => pose => class(DQ) OR  OR P=double(7,1) for pose [orientation; position] 
%                         getIK(position);       => position => double(3,1) for position-only  [x;y;z]
%                         getIK(orientation);    => orientation => double(4,1) fororientation-only (quaternion: [a;b;c;d]= a + bi + cj + dk)  
%              [jointout, output] = getIK( . );  => output is the result from IK Optimization (multistart SQP) 
% 
%     getPos
%              position = getPos(theta)          => returns the position [x;y;z] from FKM for a given joint config theta
%     getOrientation
%              orientation=getOrientation(theta) => returns the orientation (quaternion [a;b;c;d]= a + bi + cj + dk) from FKM for a given joint config theta
% 
%     getJacobGeom  
%              geomJ = getJacobGeom(theta);      => Returns geometric Jacobian (at joint configuration theta) 
% 
%     checkSelfCollision
%              boolcollision = checkSelfCollision(theta)            => Check for collision (returns true if collision is detected) and updates the pointsInArm (at joint configuration theta)  
%     getSelfCollisionPenalties
%              penalties = getSelfCollisionPenalties(theta);        => Compute penalties (internally calls checkSelfCollision) from (0 = collision) to (1 = no penalty). 
%                        = getSelfCollisionPenalties(theta,forces); => [Optional:forces] Adds forces (6xn) array of n-wrenches in task-space  
% 
%     getRandGaussianJoints
%              randnjoints = getRandGaussianJoints(); => Returns a 7-joint vector with random (Gaussian) with mean: joint-mean (between limits) and std (from joint-range) (within joint limits).
%                          = getRandGaussianJoints('seed',seedopt); => Defines the seed for the random generator (e.g., getRandGaussianJoints('seed','shuffle'))  [default: shuffle]    
%     getRandJoints
%              randjoints  = getRandJoints();         => Returns a 7-joint vector with random (Uniform) values (within joint limits).  
%                          = getRandJoints(logvec);   => [Optional:logvec] => logical(7,1) where 0 implies joint=mean(between limits) and 1 implies rand values.
%                          = getRandJoints('seed',seedopt); => [Optional:2args] Defines the seed for the random generator (e.g., getRandGaussianJoints('seed','shuffle')) [default: shuffle] 
%                          = getRandJoints('length',N);     => [Optional:2args] Using 'length' followed by N where N is the length for output random values, i.e., randjoints double(7,N)
%     plot
%               plot       = plot(joints)        => Quick plot to visualize human-arm configuration  
%
%% ====( TODO  )=============================================================
%      * Add method for elbow and shoulder FKM / Pos-Rot
%      * 
% 
%% ====( CHANGES LOG )=======================================================
%%%[ 2020-06-01 ]%%=> rHuManModel.m created
%%%[ 2020-06-29 ]%%=> class rHuManModel vs(0.3.0) published




%% DEFINE CLASS BASICS:    rHuManModel()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef rHuManModel < handle

    
%% PROPERTIES    
    properties
        kine  = []; 
        kineconfig  = []; 
        hand2tool  = DQ(1);
        % Self-Collision 
        pointsInArm = [];         
             
    end           
    
    properties (Access = private)         
        printbar = '*************************************************';
        print3   = '***';
        % CONSTANT MATRICES
        C8  = diag([1 -ones(1,3) 1 -ones(1,3)]');
        C4m = [];
        % OPenSim Extra Model
        osimmodel = []; 
        humanbox  = []; 
        humanhead = [];
        % Self-Collision 
        selfcolConfig = [];
        pos2body = [0 0 0]';
        pos2head = [0 0 0]';
        % Verbose
        verbose = false;
    end

    
%% METHODS    
    methods 
        %*************************=========================================
        %************************* MAIN CLASS
        %*************************=========================================               
        function obj = rHuManModel( options  )
            
            % Check number of extra inputs
            %===============================
            arguments
                %------------[ Human Kinematics ]
                % options.shoulderBase    {inlinemustBeA(options.shoulderBase,'DQ')}        = DQ(1);    % shoulderBase      [DQ]: DQ indicating the pose of the shoulder wrt the world
                options.dq_shoulderBase (8,1) double  = [1; zeros(7,1)];    % shoulderBase      [DQ]: DQ indicating the pose of the shoulder wrt the world
                options.shoulderHeight  (1,1) double  = 0;        % shoulderHeight    [double]: (height) indicating the position of shoulder wrt the world (in the z-axis). It will be overturned by the option shoulderBase
                options.upperArm        (1,1) double  = 0.302;    % upperArm          [double]: (length) of the upper-arm limb.
                options.foreArm         (1,1) double  = 0.2795;   % foreArm           [double]: (length) of the fore-arm limb.
                options.hand            (1,1) double  = 0.050;    % hand              [double]: (length) to the center of the hand (where forces will be applied/exerted).                 
                %------------[ Human collision assessment ]
                options.jointlimits        (7,2) double =  zeros(7,2);                
                options.joint_lowerlimits  (7,1) double =   [-90  0    -80   0        -90     0  -60]'*(pi/180);                
                options.joint_upperlimits  (7,1) double =   [130  180   +40  130      +90   +25  +60]'*(pi/180);
                    % Augmented joint limits    
                        % options.jointlimits        (7,2) double =  [[-90  0    -90   0        -90   -15  -75];[130  180   +45  145      +90   +25  +75]]'*(pi/180);                
                        % options.joint_lowerlimits  (7,2) double =   [-90  0    -90   0        -90   -15  -75]'*(pi/180);                
                        % options.joint_upperlimits  (7,2) double =   [130  180   +45  145      +90   +25  +75]'*(pi/180);                    
                %------------[ Human collision assessment ]
                options.humanbox  (3,2)  double         = [-0.3145 -0.0355; -0.065 0.065;  -0.60 0.025];  % Trunk Body as a square [xmin xmax; ymin ymax;  zmin zmax]
                options.humanheadCenter (3,1)  double   = [-0.20; 0; 0.15];   % Head as a ball (center [x;y;z])
                options.humanheadRadius (1,1)  double   = 0.14;               % Head as a ball (radius)                
     
                % Extra arguments [ verbose ]                                                                    
                options.verbose  logical  = false;     % Verbose [bool]: print outputs (particularly extras) 
            end                 
            %=================%=================%
            
                                    
            
            %===============[ Kinematics ]     
            if max(max(abs(options.jointlimits)))~=0
                options.joint_lowerlimits = options.jointlimits(:,1);
                options.joint_upperlimits = options.jointlimits(:,2);
            end
            if norm(options.dq_shoulderBase)==0,  shoulderBase = DQ(1);
            else
                shoulderBase = DQ(options.dq_shoulderBase);                
                shoulderBase = shoulderBase*(inv(norm(shoulderBase)));
            end            
            if shoulderBase == DQ(1)
                shoulderBase = DQ(1) + DQ.E*0.5*DQ([0 0 0 options.shoulderHeight]);
                obj.build_humanOpenSimKinematics(shoulderBase, options.upperArm, options.foreArm, options.hand, [options.joint_lowerlimits,options.joint_upperlimits]); 
            else
                % [obj.kine, obj.geom, obj.osimmodel] =
                obj.build_humanOpenSimKinematics(shoulderBase, options.upperArm, options.foreArm, options.hand, [options.joint_lowerlimits,options.joint_upperlimits]);   
            end
            %=================%=================%
            
            
            
            
            %=================[ Collision: Body and Head ]
            % Update humanbox and humanhead with shoulder.base
            % If there is a shoulder base
            if sum( shoulderBase.translation.q(2:4) - [0;0;0])~=0
                % If the user didnt changed the default humanbox parameters
                if max( abs( options.humanbox - [-0.35 0.00; -0.065 0.065;  -0.60 0.025] ) ) ~= 0 
                    options.humanbox = options.humanbox + shoulderBase.translation.q(2:4); 
                end
                % If the user didnt changed the default humanhead parameters
                if max( abs( options.humanheadCenter - [-0.20; 0; 0.15] ) ) ~= 0 
                    options.humanheadCenter = options.humanheadCenter + shoulderBase.translation.q(2:4); 
                end                                                    
            end
            obj.humanbox.xyz = options.humanbox;
            obj.humanbox.center = sum(obj.humanbox.xyz,2)/2;
            obj.humanbox.range  = (obj.humanbox.xyz(:,2)-obj.humanbox.xyz(:,1))/2;        
            obj.humanhead.center =  options.humanheadCenter;   
            obj.humanhead.radius =  options.humanheadRadius;           
            
            obj.selfcolConfig.armRadius_forearm  = 0.02;
            obj.selfcolConfig.armRadius_upperarm = 0.035;  
            obj.selfcolConfig.divForeArm  = 3;  % Number of divisions in the forearm 
            obj.selfcolConfig.divUpperArm = 4;  % Number of divisions in the upperarm            
            obj.selfcolConfig.armRadius = ...
                [ones(1,obj.selfcolConfig.divForeArm).*obj.selfcolConfig.armRadius_forearm ...
                 ones(1,obj.selfcolConfig.divUpperArm).*obj.selfcolConfig.armRadius_upperarm];
            obj.getSelfCollisionGains();
            %=================%=================%
            

            
            %===============[ VERBOSE ]
            obj.C4m = obj.C8(1:4,1:4);  
            obj.verbose = options.verbose;   
            printstring = strcat(10,obj.printbar,10,obj.print3,' ',obj.print3,' Loading human model...',10,obj.print3,' kinematics:');
            printstring = strcat(printstring,10,9,'upper-arm: ',32,num2str(options.upperArm));
            printstring = strcat(printstring,10,9,'fore-arm: ',32, num2str(options.foreArm));
            printstring = strcat(printstring,10,9,'point of force at hand: ',32,num2str(options.hand));
            printstring = strcat(printstring,10,9,'shoulder-base position (xyz): [',num2str(shoulderBase.translation.q(2:4)'),']');
            printstring = strcat(printstring,10,9,'shoulder-base orientation (quat): [',num2str(shoulderBase.P.q(1:4)'),']');            
            printstring = strcat(printstring,10,obj.print3,' Body Geometry for self collision');
            printstring = strcat(printstring,10,9,'Body 3D-box: x \in [',num2str(obj.humanbox.xyz(1,:)),'], y \in [',num2str(obj.humanbox.xyz(2,:)),'], z \in [',num2str(obj.humanbox.xyz(3,:)),'] ');
            printstring = strcat(printstring,10,9,'Head 3D-ball with radius: [',num2str(obj.humanhead.radius),'] centered at [',num2str(obj.humanhead.center'),'] ');
            printstring = strcat(printstring,10,9,'Upper-arm Cylinder-Radius: [',num2str(obj.selfcolConfig.armRadius_upperarm),'] with [',num2str(obj.selfcolConfig.divUpperArm'),'] discrete equally spaced points.');
            printstring = strcat(printstring,10,9,'Fore-arm Cylinder-Radius: [',num2str(obj.selfcolConfig.armRadius_forearm),'] with [',num2str(obj.selfcolConfig.divForeArm'),'] discrete equally spaced points.');
            printstring = strcat(printstring,10,obj.print3,' Definitions');
            printstring = strcat(printstring,10,9,'Shoulder orientation: (+x) lateral towards right, (+y) front, (+z) upwards');
            printstring = strcat(printstring,10,9,'Wrist orientation: (+x) Palm-in, (+y) Thumbs-ub, (+z) defined from wrist to fingers',10);
            obj.printv( obj,printstring );
            %=================%=================%
                        
        end

        
        
        
        
    %*************************=========================================
    %************************* PUBLIC METHODS    
    %*************************=========================================                             
        geomJ = getJacobGeom(obj, theta7);
        % Get Inverse Kinematics 
        [jointout, output] = getIK(obj, pose);
        % Get Forward Kinematics
        poseout = getFKM(obj, theta7);
        % Get Position from Forward Kinematics
        positionout = getPos(obj, theta7);
        % Get Orientation from Forward Kinematics
        rotationout = getOrientation(obj, theta7);  
        % Update points in arm
        boolcollision = checkSelfCollision(obj,theta7);
        % Compute penalties
        penalties = getSelfCollisionPenalties(obj,theta7,forces);
        %
        %
        % Get random (uniform) joints (within limits). Optional logical
        % (7,1) where 0 implies joint=0 and 1 implies rand. 
        jointsout = getRandJoints(obj,logtheta7, options);
        % Get random (Gaussian) joints (within limits).
        jointsout = getRandGaussianJoints(obj, options)
        
        function plot(obj,joints)             
            obj.kine.plot( obj.getJoints8(joints) )
        end
    end
    
        
    
    %*************************=========================================
    %************************* PROTECTED
    %*************************=========================================                                 
    methods (Access = protected)     
        % Build human kinematics from opensim model
        build_humanOpenSimKinematics(obj, shoulderPose, length_upperarm, length_forearm, length_hand, joint_limits);        
        % Get Self Collision Gains
        getSelfCollisionGains(obj)
    end
    
    
    %*************************=========================================
    %************************* STATIC INTERN METHODS:
    %*************************=========================================             
    methods (Static, Access = protected)  
        % Custom validator functions
        mustBeA(input,className)        
        % Print data only if validation is on (verbose)
        printv(obj,string)
        
        % Get joints8 from joints7 data
        function joints8out = getJoints8(theta7) 
            joints8out = [theta7(1);  theta7(2);  theta7(1);  -theta7(3);    theta7(4); theta7(5); theta7(6); theta7(7);  ];
        end
        %
            
           
        
    end            
        

end

%%




