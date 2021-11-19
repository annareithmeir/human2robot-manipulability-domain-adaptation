% h = plot(dq,OPTIONS) plots the dual quaternion dq and returns the handle of the plot. 
% The following options are available:
%   'scale'
%   'erase'
% Ex.: plot(dq,'scale',5) will plot dq with the axis scaled by a factor of 5
%      
% plot(dq,'erase', h) will plot dq, but erasing previous values of dq. The parameter h
% is the handle of the plot. This is useful for plotting frames in motion
% (if erase is not used, it will leave a trail)
function handle=plot(dq,varargin)

if norm(dq) ~= 1
    error('Only unit dual quaternions can be plotted');
end

optargin = size(varargin,2);

scale = 1;
lineWidth = 1;
rotFollowedByTrans = false;    
blackwhite          = false; 
printtext = true;                    

if optargin > 0
    for j =1:optargin
        
        my_string = varargin(1,j);%num2str(cell2mat(varargin(1,j)));
       
        if(~isstruct(my_string{1}))
            if strfind(my_string{1},'era')
                handle_cell = varargin(1,j+1);  
                handle = handle_cell{1};
                    for i = 1:3   
                        delete(handle.handle_axis{i});
                        delete(handle.handle_text{i});
                    end 
                   
            elseif strfind(my_string{1},'sca')
                   scale = cell2mat(varargin(1,j+1));              
                   
            elseif strfind(my_string{1},'lineWi')
                lineWidth = cell2mat(varargin(1,j+1));    
                
            elseif strfind(my_string{1},'altpos')
                rotFollowedByTrans = true;               
            
            elseif strfind(my_string{1},'bw')
                blackwhite = true;    
            elseif strfind(my_string{1},'notext')
                printtext = false;                    
            end            
        end
    end
end

if (blackwhite)
    printcolors = ['k', 'k', 'k'];
else
    printcolors = ['r', 'g', 'b'];
end
    
    
    % create unit vectors and rotate them by the quaternion part of dq.
    t1 = 1+DQ.E*[0, scale*0.5, 0, 0];
    t2 = 1+DQ.E*[0, 0, scale*0.5, 0];
    t3 = 1+DQ.E*[0, 0, 0, scale*0.5];
    
    %Do a vector rotation
	x = dq.P*t1*dq.P';
	y = dq.P*t2*dq.P';
	z = dq.P*t3*dq.P';
    
    old_ishold = ishold;
    
    if rotFollowedByTrans
        dq_t = dq.P'*translation(dq)*dq.P;
        x_t = translation(x);
        y_t = translation(y);
        z_t = translation(z);
    else
        dq_t = translation(dq);
        x_t = translation(x);
        y_t = translation(y);
        z_t = translation(z);
    end
    
        
    
%     handle.handle_axis{1} = plot3([0;x_t.q(2)]+dq_t.q(2), [0; x_t.q(3)]+dq_t.q(3), [0; x_t.q(4)]+dq_t.q(4), 'r','Linewidth',1,'erasemode', 'xor');
    handle.handle_axis{1} = plot3([0;x_t.q(2)]+dq_t.q(2), [0; x_t.q(3)]+dq_t.q(3), [0; x_t.q(4)]+dq_t.q(4), printcolors(1),'Linewidth',lineWidth);
    if(~ishold)
        hold on;
    end    
    if printtext
        handle.handle_text{1} = text(dq_t.q(2)+x_t.q(2), dq_t.q(3)+x_t.q(3), dq_t.q(4)+x_t.q(4), 'x');
        set(handle.handle_text{1} , 'Color', 'k');
    end
    
% 	handle.handle_axis{2} =plot3([0;y_t.q(2)]+dq_t.q(2), [0; y_t.q(3)]+dq_t.q(3), [0; y_t.q(4)]+dq_t.q(4), 'g','Linewidth',1,'erasemode', 'xor');
    handle.handle_axis{2} =plot3([0;y_t.q(2)]+dq_t.q(2), [0; y_t.q(3)]+dq_t.q(3), [0; y_t.q(4)]+dq_t.q(4), printcolors(2),'Linewidth',lineWidth);
    if printtext
        handle.handle_text{2}  = text(dq_t.q(2)+y_t.q(2), dq_t.q(3)+y_t.q(3), dq_t.q(4)+y_t.q(4), 'y');
        set(handle.handle_text{2} , 'Color', 'k');
    end
% 	handle.handle_axis{3} = plot3([0;z_t.q(2)]+dq_t.q(2), [0; z_t.q(3)]+dq_t.q(3), [0; z_t.q(4)]+dq_t.q(4), 'b','Linewidth',1,'erasemode', 'xor');
    handle.handle_axis{3} = plot3([0;z_t.q(2)]+dq_t.q(2), [0; z_t.q(3)]+dq_t.q(3), [0; z_t.q(4)]+dq_t.q(4), printcolors(3),'Linewidth',lineWidth);    
    if printtext
        handle.handle_text{3}  = text(dq_t.q(2)+z_t.q(2), dq_t.q(3)+z_t.q(3), dq_t.q(4)+z_t.q(4), 'z');
        set(handle.handle_text{3} , 'Color', 'k');
    end
    if(~old_ishold)
        hold off;
    end
    
end