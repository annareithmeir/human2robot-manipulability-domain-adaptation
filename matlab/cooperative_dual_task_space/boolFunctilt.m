function bv = boolFunctilt(da, tiltang, tilt_min, tilt_max)

if tiltang>tilt_min && tiltang < tilt_max
    bv = true;
% elseif tiltang <= tilt_min && da > 0
%     bv = true;
% % elseif tiltang <= tilt_min && da < 0
% %     bv = false;
% elseif tiltang >= tilt_max && da < 0
%     bv = true;
else
    bv = false;


end