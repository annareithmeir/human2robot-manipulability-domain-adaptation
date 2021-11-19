%% Custom validator functions
function mustBeA(input,className)
    % Test for specific class    
    if ~isa(input,className)
        error([10, strcat('*** Input must be of class: ',9,className)])
    end
end    