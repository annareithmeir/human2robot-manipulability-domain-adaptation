
%
% Luis - Complement to get geometric jacobian
% Inputs: dq_kinematics and joint configuration
% Output: 6xn geometric jacobian
%
function [J, J8, J6] = geomJ(xm,Jacob)
%
C8 = diag([-1 ones(1,3) -1 ones(1,3)]');
C4m = -C8(1:4,1:4);  
CJ4_2_J3= [zeros(3,1) eye(3)];
   
%    if ismethod(kine,'pose_jacobian')
%        Jacob = kine.pose_jacobian(q);
%        xm = kine.fkm(q);
%    elseif ismethod(kine,'jacobian')        
%        Jacob = kine.jacobian(q);
%        xm = kine.fkm(q);
%    elseif strcmpi( class(kine), 'double') && strcmpi( class(q), 'DQ')
%        xm = q;
%        Jacob = kine;
%    end

    J8(1:4,:) = 2*haminus4(xm.P')*Jacob(1:4,:);
    J8(5:8,:) = 2*( hamiplus4(xm.D)*C4m*Jacob(1:4,:) +  haminus4(xm.P')*Jacob(5:8,:)  );    
%    
%     J(1:3,:) = CJ4_2_J3*2*haminus4(xm.P')*Jacob(1:4,:);
%     J(4:6,:) = CJ4_2_J3*2*( hamiplus4(xm.D)*C4m*Jacob(1:4,:) +  haminus4(xm.P')*Jacob(5:8,:)  );
%    
    J(1:3,:) = CJ4_2_J3*J8(1:4,:);
    J(4:6,:) = CJ4_2_J3*J8(5:8,:);    
    J6(1:3,:) = J(1:3,:);
    J6(4:6,:) = 2*J(4:6,:);
   
end
