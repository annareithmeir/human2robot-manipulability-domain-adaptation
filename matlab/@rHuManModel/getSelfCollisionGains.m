%% static function returning self collision gains
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
% 
%  Read referenced paper for further details
% 
%-------------------------------------------------------------------------- 
%   ====( CHANGES LOG )====         
%-------------------------------------------------------------------------- 



%% Function load (outputs) kinematics 
% 
function getSelfCollisionGains(obj)
    % Compute the cost variables for self-collision
    obj.selfcolConfig.distgains.body = -log(  (1-0.8)/0.8 *0.05^1)/0.05;  % dist 0.05 (5cm) with cost reduction of 0.8  (more linear)
    obj.selfcolConfig.distgains.head = -log(  (1-0.7)/0.7 *0.10^2)/0.10;  % dist 0.10 (10cm) with cost reduction of 0.7 (less linear)

    obj.selfcolConfig.distgains.alpha_body = obj.selfcolConfig.distgains.body;
    obj.selfcolConfig.distgains.alpha_head = obj.selfcolConfig.distgains.head;
    obj.selfcolConfig.distgains.beta_body  = 1;
    obj.selfcolConfig.distgains.beta_head  = 2;             
end

