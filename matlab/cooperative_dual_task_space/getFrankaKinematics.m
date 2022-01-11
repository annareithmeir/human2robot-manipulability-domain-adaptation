
        
function kin = getFrankaKinematics(obj)
    %Standard D-H of FE Panda
%             FEp_DH_theta  = [0, 0, 0, 0, 0, 0, 0];
%             FEp_DH_d      = [0.333, 0, 0.316, 0, 0.384, 0, 0];
%             FEp_DH_a      = [0, 0, 0, 0.0825, -0.0825, 0, 0.088];
%             FEp_DH_alpha  = [0, -pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2];
%             FEp_DH_matrix = [FEp_DH_theta;
%                             FEp_DH_d;
%                             FEp_DH_a;
%                             FEp_DH_alpha];

    FEp_DH_theta  = [0,     0,      0,      0,        0,        0,      0];
    FEp_DH_d      = [0.333, 0,      0.316,  0,        0.384,    0,      0.107];
    FEp_DH_a      = [0,     0,      0.0825, -0.0825,  0,        0.088   0.0003];
    FEp_DH_alpha  = [-pi/2, pi/2,   pi/2,   -pi/2,    pi/2,     pi/2    0];

    FEp_DH_matrix = [FEp_DH_theta;
                    FEp_DH_d;
                    FEp_DH_a;
                    FEp_DH_alpha];


    kin = DQ_SerialManipulator(FEp_DH_matrix,'standard');
    % We set the transformation from the world frame to the robot
    % base frame. Therefore, the end-effector pose is given by
    % pose_effector = transformation_from_world_to_base*fkm(q);
    if exist('obj','var')
        kin.set_reference_frame(obj.vrep_interface.get_object_pose(obj.base_frame_name));
        kin.set_base_frame(obj.vrep_interface.get_object_pose(obj.base_frame_name));
    %             kin.set_effector(1+0.5*DQ.E*DQ.k*0.1070);      
        
    end
end