% CLASS DQ_cdts
% Usage: cdts = DQ_cdts(robot1,base1,robot2,base2), where:
% - robot1 and robot2 are objects of type DQ_kinematics;
% - base1 and base2 are dual quaternions representing the pose of each arm
%   with respect to the world.
%
% By using this class, the cooperative system is described by the
% cooperative variables xa and xr, and their respective Jacobians Ja and Jr
%
% Type DQ_cdts.(method or property) for specific help.
% Ex.: help DQ_cdts.xa
%
% METHODS:
%       xa
%       xr
%       Ja
%       Jr
%       x1
%       x2
%


classdef DQ_cdts
    properties
        robot1, robot2;
    end
    
    methods
        function obj = DQ_cdts(robot1, robot2)
            if ~isa(robot1,'DQ_kinematics') || ~isa(robot2,'DQ_kinematics')
                error('The DQ_cdts class must be initialized with the kinematics information of each robot');
            else
                obj.robot1 = robot1;
                obj.robot2 = robot2;
            end
        end
        
        function x = x1(obj,theta)
            % x = x1(theta) returns the pose of the first arm,
            % where theta is the joint position of the resultant system;
            % that is, theta = [theta1;theta2]
            theta1=theta(1:obj.robot1.links-obj.robot1.n_dummy);
            x = obj.robot1.fkm(theta1);
        end
        
        function x = x2(obj,theta)
            % x = x2(theta) returns the poqz of the second arm,
            % where theta is the joint position of the resultant system;
            % that is, theta = [theta1;theta2]           
            theta2=theta(obj.robot1.links-obj.robot1.n_dummy+1:end);
            x = obj.robot2.fkm(theta2);
        end
        
        function J = jacobian1(obj,theta)
            % J = jacobian1(theta) returns the jacobian of the first arm,
            % where theta is the joint position of the resultant system;
            % that is, theta = [theta1;theta2]
            theta1=theta(1:obj.robot1.links-obj.robot1.n_dummy);
            J = obj.robot1.jacobian(theta1);
        end
        
        function J = jacobian2(obj,theta)
            % J = jacobian2(theta) returns the jacobian of the second arm,
            % where theta is the joint position of the resultant system;
            % that is, theta = [theta1;theta2]
            theta2=theta(obj.robot1.links-obj.robot1.n_dummy+1:end);
            J = obj.robot2.jacobian(theta2);
        end
        
        function x = xr(obj,theta)
            % x = xr(theta) returns the relative dual position,
            % where theta is the joint position of the resultant system;
            % that is, theta = [theta1;theta2]
                       
            x = obj.x2(theta)'*obj.x1(theta);
        end
        
        function x = xa(obj,theta)
            % x = xr(theta) returns the absolute dual position,
            % where theta is the joint position of the resultant system;
            % that is, theta = [theta1;theta2]            
            x = obj.x2(theta)*(obj.xr(theta)^0.5);
        end
        
        function jac = Jr(obj,theta)
            % x = xr(theta) returns the relative Jacobian,
            % where theta is the joint position of the resultant system;
            % that is, theta = [theta1;theta2]
            jac = [hamiplus8(obj.x2(theta)')*obj.jacobian1(theta), haminus8(obj.x1(theta))*DQ_kinematics.C8*obj.jacobian2(theta)];
        end
        
        function jac = Ja(obj, theta)
            % x = xr(theta) returns the absolute Jacobian,
            % where theta is the joint position of the resultant system;
            % that is, theta = [theta1;theta2]
            x2 = obj.x2(theta);
            
            jacob2 = obj.jacobian2(theta);            
            jacobr = obj.Jr(theta);
            xr=obj.xr(theta);
            
            jacob_r2 = 0.5*haminus4(xr.P'*(xr.P)^0.5)*jacobr(1:4,:);
            jacobp_r = DQ_kinematics.jacobp(jacobr,xr);
            
            jacob_xr_2 = [jacob_r2; 0.25*(haminus4(xr.P^0.5)*jacobp_r+hamiplus4(translation(xr))*jacob_r2)];
            
            jac = haminus8(xr^0.5)*[zeros(8,obj.robot1.links-obj.robot1.n_dummy),jacob2]+hamiplus8(x2)*jacob_xr_2;
        end
    end
end