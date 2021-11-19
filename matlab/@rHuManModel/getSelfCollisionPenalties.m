%% get self collision penalties
% 
%--------------------------------------------------------------------------
%  This method is part of the class rHuManModel
%  -
%  Compute penalties (internally calls checkSelfCollision)
%  Read referenced paper for further details on details of how it works
%  
%-------------------------------------------------------------------------- 
% Function inputs:
%     theta             -  double(7,1) with joint values
%     [Optional:forces] -  Adds forces (6xn) array of n-wrenches in task-space  
% 
% Function outputs:
%       penaltOut  -  Returns penalty value (for single input or single force) 
%                  -  Returns penalty vector for multiple-forces (1 for each)
%                  -  Returns 0 or (vector of zeros) if self-collision is detected.

%-------------------------------------------------------------------------- 
% 
%      penalty   = getSelfCollisionPenalties(theta);        =>  penalty value 
%      penalties = getSelfCollisionPenalties(theta,forces); =>  penalties (1 for each force) 
% 
%-------------------------------------------------------------------------- 

%% Function getSelfCollisionPenalties
% 
function penaltOut = getSelfCollisionPenalties(obj,theta7,forces)
    FLAG_MULTIPLE = false;
    % Definining if we are checking against a set of forces
    if exist('forces','var')
        lengthforces = size(forces,2); 
        penaltOut = ones(1, lengthforces ); 
        FLAG_MULTIPLE = true;
    % or regardless the force    
    else
        penaltOut = 1;
    end
    % If there is a collision check then penalt=0.
    if obj.checkSelfCollision(theta7)
       penaltOut = 0.*penaltOut;
       return 
    end
    %

    dist2Body = norm( obj.pos2body ); 
    dist2Head = norm( obj.pos2head );  
    penaltBody = 1/(1+exp(-(obj.selfcolConfig.distgains.body)*dist2Body)*dist2Body^-2);
    penaltHead = 1/(1+exp(-(obj.selfcolConfig.distgains.head)*dist2Head)*dist2Head^-2);

    if FLAG_MULTIPLE
        % Go through all forces (3D force)
        for i=1:lengthforces
            if norm( forces(4:6,i) )==0
                penaltOut(i) = penaltHead*penaltBody; 
            else 
                if forces(4:6,i)'*obj.pos2body > 0
                    penaltOut(i) = penaltHead*penaltBody; 
                else
                    penaltOut(i) = penaltHead; 
                end                
            end
        end  
    %==============    
    else
        % penalt = min(1,max(0,min(penaltBody, penaltHead) ));
        penaltOut = penaltHead*penaltBody;                                 
    end
end
