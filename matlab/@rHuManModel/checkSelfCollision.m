%% update points in the arm (for collision detection) 
% 
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
%  Divides the arm cylinders in different balls and check distance to balls
%  if it respects a given radius then there is no collision
% 
%-------------------------------------------------------------------------- 
%   [collisiondetected] = checkSelfCollision(theta7) 
%-------------------------------------------------------------------------- 



%% Function load (outputs) kinematics 
% 
function [collisiondetected] = checkSelfCollision(obj,theta7)
        collisiondetected = false; 
% function updatePointsInArm(obj,theta7)

        %% Update points in the arm
        xm_elbow = obj.kine.fkm( obj.getJoints8(theta7),4);
        xm_wrist = obj.kine.fkm( obj.getJoints8(theta7),8);  
                
        elbowpos = xm_elbow.translation.q(2:4);
        wristpos = xm_wrist.translation.q(2:4);
        shoulderpos = obj.kineconfig.shoulderPose.translation.q(2:4);
        
        upperarmDist = elbowpos - shoulderpos;
        forearmDist  = wristpos - elbowpos;
        for i=1:obj.selfcolConfig.divForeArm,    obj.pointsInArm{i}                              = shoulderpos + upperarmDist*i/(obj.selfcolConfig.divForeArm); end
        for i=1:obj.selfcolConfig.divUpperArm,   obj.pointsInArm{i+obj.selfcolConfig.divForeArm} = elbowpos + forearmDist*i/(obj.selfcolConfig.divUpperArm);     end               
        
        
        %% Compute distance to body 
        
        pos = obj.getFKM(theta7).translation.q(2:4);                
        
        % Translation from body to end-effector 
        distVector = pos - obj.humanbox.center;
        
        % Set distance to zero (if inside body) 
        % Invert vector translation: now it is end-effector to body 
        obj.pos2body = min( 0 , obj.humanbox.range - abs(distVector) ).*( sign( distVector - obj.humanbox.range ) );         
        
        %================================ Check for collision
        if norm( obj.pos2body )==0            
            obj.pos2body= [0 0 0]';
            collisiondetected = true;             
        end
                   
        % Check if any of the points in the arm are inside body                      
        for i=1:(obj.selfcolConfig.divForeArm + obj.selfcolConfig.divUpperArm)
           distVector = obj.humanbox.center - obj.pointsInArm{i};
           if norm( min( 0 , obj.humanbox.range - abs(distVector) ) ) <= obj.selfcolConfig.armRadius(i)
               obj.pos2body= [0 0 0]';
               collisiondetected = true; 
           end
        end      
        
        %% Compute distance to head
        
        % Vector translation from end-effector to HEAD 
        distVector = obj.humanhead.center - pos;
        
        % Distance from end-effector to HEAD 
        distance2head = max(0, norm( distVector ) - obj.humanhead.radius); 
        
        % Check for collision
        if distance2head==0
            obj.pos2head = [0 0 0]';
            collisiondetected = true;             
        end        
        
        % Get translation to HEAD discounted the head radius
        obj.pos2head = distVector.*( distance2head/norm(distVector) );        
        
        
        %================================ Check for collision                 
        % Check if any of the points in the arm are inside body        
        for i=1:(obj.selfcolConfig.divForeArm + obj.selfcolConfig.divUpperArm)
           distVector = obj.humanhead.center - obj.pointsInArm{i};
           distance2head = max(0, norm( distVector ) - obj.humanhead.radius); 
           if distance2head <= obj.selfcolConfig.armRadius(i)
               obj.pos2head = [0 0 0]';               
               collisiondetected = true; 
           end
        end   
        
       
                 
        
end

