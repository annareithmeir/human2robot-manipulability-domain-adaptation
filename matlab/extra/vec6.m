% The function vec6 returns 6D vector representation from the pure dual
% quaternion which is obtained from the vector isomorphism.
% Also outputs the inverse mapping if the input is a vector, in this case,
% it returns a pure dual quaternion
% 
function v = vec6(dq_vec)
    % if class(dq_vec)=='DQ' would do a better job (but is slower) |
    % numeric is even slower
    if length(dq_vec)==1
        % Return the 6D vector representation from the pure quaternion
        % (from vector isomorphism) 
        v = dq_vec.q( [2:4,6:8] );        
        
    else        
        % Return the pure quat from the 3D vector representation 
        v = DQ([0; dq_vec(1:3); 0; dq_vec(4:6)]);  
    end
end