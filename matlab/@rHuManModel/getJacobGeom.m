%% return geometric Jacobian
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
% Compute geometric Jacobian from joint inputs
% 
%--------------------------------------------------------------------------
% Function inputs:
%       joints     -  (7,1) joint vector
% 
% Function outputs:
%       geometric jacobian (6,7) matrix
% 
%-------------------------------------------------------------------------- 
% 
% J =  getJacobGeom(theta)
% 
%-------------------------------------------------------------------------- 



%% Function load (outputs) kinematics 
% 
function [geomJ] = getJacobGeom(obj, theta7)
    
    %joints8 = [theta7(1);  theta7(2);  theta7(1);  -theta7(3);    theta7(4); theta7(5); theta7(6); theta7(7);  ];
    joints8 = obj.getJoints8(theta7);
    xm = obj.kine.fkm( joints8 );
    jacobTmp = obj.kine.jacobian( joints8  );
    jacob = [(jacobTmp(:,1)+jacobTmp(:,3)) jacobTmp(:,2) -jacobTmp(:,4)  jacobTmp(:,5:8) ];
    
    % Transformation

%     G1 = [2*haminus4( xm.P' )           zeros(4,4)];
%     G2 = [2*obj.C4m*haminus4( xm.D' )   2*haminus4( xm.P' )];
%     jacobGeom = [G1; G2]*jacob;
    
    m1 = 2*haminus4( xm.P' );    
    G2 = [2*obj.C4m*haminus4( xm.D' )   m1];   
    jacobGeom = [m1*jacob(1:4,:);  G2*jacob ];
    
    % Output Geometric Jacobian 6 x n
    geomJ = jacobGeom([2:4,6:8],:);                            

end

