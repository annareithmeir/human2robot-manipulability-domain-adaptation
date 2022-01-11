% Script for dynamic control of FE Panda Arms in the Cooperative Dual Space

clear;
close all;
clc;

include_namespace_dq; 

%% Initialize VREP Robots
% fep_vreprobot1 = FEpVrepRobot1('Franka1',vi);
% 
% fep_vreprobot2 = FEpVrepRobot2('Franka2',vi);
% 
% 
% % Robot1 
% kin.set_reference_frame(obj.vrep_interface.get_object_pose(obj.base_frame_name));
% kin.set_base_frame(obj.vrep_interface.get_object_pose(obj.base_frame_name));  

fep1 = getFrankaKinematics();
fep2 = getFrankaKinematics();
baseR1 = DQ([0.887010404008924;0;0;0.461749437660648;-2.236195910568693e-08;-0.067005303402133;0.702186583093093;4.295682628495998e-08]);
baseR2 = DQ([0.499999888056226;0;0;-0.866025468415197;5.484533668703717e-08;-0.384077708534888;-0.650989969524217;3.166496044753432e-08]);

baseR1 = baseR1*inv(norm(baseR1));
fep1.set_reference_frame(baseR1);
fep1.set_base_frame(baseR1);
fep2.set_reference_frame(baseR2);
fep2.set_base_frame(baseR2);

% Default Start
q1 = [2.722713708877564;-0.209439516067505;0.261799395084381;-1.570796370506287;0;2.792526721954346;1.570796370506287];
q2 = [2.722713708877564;-0.209439516067505;0.436332315206528;-1.570796370506287;0;2.792526721954346;1.570796370506287];
% q = [q1;q2];

dq_init_1 = fep1.fkm(q1);
dq_init_2 = fep2.fkm(q2);

%% Load DQ Robotics kinematics
% fep1  = fep_vreprobot1.kinematics();
% fep2  = fep_vreprobot2.kinematics();

panda_bimanual = DQ_CooperativeDualTaskSpace(fep1,fep2);

% maximum joint ranges (deg): (q1..q7)
%       -166.0031 -101.0010 -166.0031 -176.0012 -166.0031  -1.0027  -166.0031
q_min = [-2.8973   -1.7628   -2.8973   -3.0718   -2.8973   -0.0175  -2.8973];

%        166.0031  101.0010  166.0031 -3.9992   166.0031   215.0024  166.0031
q_max = [ 2.8973    1.7628    2.8973  -0.0698    2.8973    3.7525    2.8973];

% Finding the centre of the joint ranges
q_c = 0.5 * (q_max + q_min);

% Total range of the joints (in radians)
del_q = q_max - q_min;

%% Matlab Plot
FLAGPLOT =false;


if FLAGPLOT 
    figX = figure;
    axis equal;
    plot(fep1, q1);
    plot(fep2, q2);    

    grid on;
    view(-0,0)
    hold on;

    plot(fep1, q1);
    plot(fep2, q2);
    
    axis([-1.8   -0.5   0.0    1.5    0.0000    0.8])
    view(10,15)
    drawnow;
pause    
end



%% Get Joint Handles

% handles = get_joint_handles(vi,vi.clientID);
% joint_handles1 = handles.armJoints1;
% joint_handles2 = handles.armJoints2;

% get initial state of the robot (Using VRep Matlab Remote API directly)
qstr = '[ ';
qdotstr = '[ ';


% Default Start
% q1 = [2.722713708877564;-0.209439516067505;0.261799395084381;-1.570796370506287;0;2.792526721954346;1.570796370506287];
% q2 = [2.722713708877564;-0.209439516067505;0.436332315206528;-1.570796370506287;0;2.792526721954346;1.570796370506287];
q = [q1;q2];
dq = zeros(14,1);

% Bottom start
% q1 = [2.733342497539070;-0.306707085342838;0.265933667484359;-2.185896738255447;0.025823042937157;3.312891331100007;1.572010724424123];
% q2 = [2.732030426598679;-0.304976750793902;0.443931879400576;-2.177188003178219;0.003210408946967;3.311546970957573;1.609428367428486] ;


%% Field based Motion planning final condition and parameters

q1_goal = [2.4384 -0.3518 0.5410 -1.9826 0.3599 3.1207 1.3489];
dq_goal_1 = fep1.fkm(q1_goal);
dq_goal_o_1 = dq_goal_1.rotation;

dqrd = panda_bimanual.relative_pose(q);
dqad_ant = panda_bimanual.absolute_pose(q);

p_init_1 = dqad_ant.translation.q(2:4);
% p_init_2 = dq_init_2.translation.q(2:4);

ini_o_1 = dqad_ant.P;

p_goal_1 = dq_goal_1.translation.q(2:4);

goal = p_goal_1';
Obstacles1 = 0.5*( p_goal_1' + p_init_1' );
Obstacles2 = [-1.146 0.7606 0.8609]; 
Obstacles3 = [-1.115 0.7918 0.8194]; 
Obstacles4 = [-1.326 0.6618 0.8034]; 
Obstacles5 = [-1.576 0.7638 0.8534]; 
obs_pos = [Obstacles1; Obstacles2; Obstacles3];
obs_rad = 0.01;

k_circ = 0.1;
k_attr = 0.2;
robot_pos = p_init_1';
robot_vel = zeros(1,3);

% Parameters
ROBOT_MASS = 1.0;
time_step = 0.01;
V_MAX = 0.05;

% [x,xdot, xddot, y,ydot, yddot, z, zdot, zddot,t] = ...
%     circularFields(robot_pos, robot_vel, goal, obs_pos, obs_rad, k_attr, k_circ);

%% Task definitions

% dqad = DQ([cos(pi/32);0;sin(pi/32);0]) .* dqad_ant;
dqad = ( DQ(1) + DQ.E*0.5*DQ([0 0 0 -0.15])*DQ(1) ) .* dqad_ant;  


task_d = [dqad.q;dqrd.q];
task_d2 = dqad.q;
task_d1 = dqrd.q;


% Sampling time 
% sampling_time = 0.005; % 
sampling_time = 0.010; % V-REP's sampling time 

%% Defining Constraints based on Primitives

% Current Orientation
r = fep1.fkm(q(1:7)).P;

% Desired Plücker line in k_
l = ini_o_1 * k_ * ini_o_1';


%% Data for analyzing later
data.fep_xd = [];
data.fep_xm = [];
data.fep_q_1 = [];
data.fep_q_2 = [];
data.cf_force = [];
data.min_svd = [];
data.fep_abs_error_norm = [];
data.fep_rel_error_norm = [];
data.fep_ee_pos_error_norm = [];

%% Two-arm control
epsilon = 0.001; %for the stop condition
error = epsilon+1;
i = 0;
iter = 1;
jj=1;
% 

% Starting min singular value along the trajectory
min_sv = 0.38; % Checked from the starting values of the initial trajectory

pos_err = p_goal_1 - p_init_1;
d = norm(goal - robot_pos);
tic
% %The sweep motion (back and forth) will be performed twice
% while norm(pos_err) > 0.1 % Condition when using CFs
while norm(pos_err) > 0.01
    if ~mod(jj,100)
        disp(jj)
    end

    q = [q1;q2];

    % Getting current pose and Jacobian
    jacob1 = panda_bimanual.relative_pose_jacobian(q);
    geomJac = geomJ(panda_bimanual.absolute_pose_jacobian(q), panda_bimanual.absolute_pose(q));
    %     geomJac2 = geomJ(fep2,q(8:14));
    jacob2 = panda_bimanual.absolute_pose_jacobian(q);
    taskm1 = vec8(panda_bimanual.relative_pose(q));
    taskm2 = vec8(panda_bimanual.absolute_pose(q));  

    % Translation Jacobian for the absolute pose
    J_t = geomJac(4:6,:);

    % SVD of the Translation Jacobian to get the principal directions
    [U, S, V] = svd(J_t);

    if S(3,3) < min_sv
        min_sv = S(3,3);
    end

    %% Circular Fields

    % reseting the forces so that residuals are removed
    force = zeros(1,3);

    % Add the attractor forces
    att_force = AttractorForces(goal, robot_pos, k_attr);
    force = force + att_force;

    for j = 1:size(obs_pos,1)

        % Add circular forces for each obstacle
        circ_force = CurrForces(obs_pos(j,:), obs_rad, robot_pos, robot_vel, goal, k_circ);

        force = force + circ_force;

    end
    
    % Additional repulsive force for increasing manipulability
    if S(3,3) < 0.33
        
        if dot(U(:,3),force) >= 0
            
            fu = -0.2 * (1 - 0.23/S(3,3)) * U(:,3);
        
        else
            fu = 0.2 * (1 - 0.23/S(3,3)) * U(:,3);
        end
        
        force = force + fu';
    end

    acc = force / ROBOT_MASS;

    robot_acc = acc;
    robot_vel = robot_vel + acc * time_step;
    vel_norm = norm(robot_vel);

    if vel_norm > V_MAX
        robot_vel = robot_vel * (V_MAX/vel_norm); 
    end

    robot_pos = robot_pos + 0.5 * acc * time_step * time_step+ robot_vel * time_step;

    % If the robot is too close to the goal slow it down
    if d < 0.1
        robot_vel = robot_vel * (10*d);
    end  

    % Update the distance to goal
    d = norm(goal - robot_pos);

    x(jj) = robot_pos(1);
    xdot(jj) = robot_vel(1);
    xddot(jj) = robot_acc(1);
    y(jj) = robot_pos(2);
    ydot(jj) = robot_vel(2);
    yddot(jj) = robot_acc(2);
    z(jj) = robot_pos(3);
    zdot(jj) = robot_vel(3);
    zddot(jj) = robot_acc(3);
    t(jj) = jj * time_step;

    % Desired Trajectory (Taking points one by one from the CF)
    p = x(jj)*i_ + y(jj)*j_ + z(jj)*k_;
    task_dq = ini_o_1 + DQ.E * 0.5 * p * ini_o_1;
    task_d_abs = vec8(task_dq); 


    % Getting current error
    error  = task_d1-taskm1;
    error2 = task_d_abs-taskm2;

    % PseudoInverse
    robustInv1 = jacob1'*pinv(jacob1*jacob1' + 0.001*eye(8) );
    robustInv2 = jacob2'*pinv(jacob2*jacob2' + 0.001*eye(8) );    
    %     robustInv1 = pinv(jacob1);    

    % Rotation Jacobian
    Jr_r = jacob1(1:4,:);

    % Current Orientation
    r = fep1.fkm(q(1:7)).P;

    % Current Plücker line in k_
    lz = r * k_ * r';

    % Line-static-Line angle Jacobian
    J_rz = haminus4(k_ * r')* Jr_r + hamiplus4(r*k_)*dqrd.C4 * Jr_r;

    % Norm of the error
    e_n = norm(l - lz);

    % Error
    err_t = vec4( l - lz);

    % Line Jacobian error def
    J_lerr = -2 * err_t' * J_rz;
    robustJ_lerr = J_lerr' * pinv(J_lerr * J_lerr' + 0.001 * eye(1));

    % Main controller
    %     dq = 100*robustInv1*0.5*error;

    % Gain 
    gain = 100;

    % Subtask
    Pjacob1 = eye(14) - pinv(jacob1)*jacob1;     
    %     dq = dq + Pjacob1*( gain*robustInv2*error2   ); 


    % joint1 avoid
    jacobAug = [jacob1; J_lerr; jacob2]; 
    Pjacobaug = eye(14) - pinv(jacobAug)*jacobAug;     
    %     dq = dq + Pjacob1*( 100*robustInv2*error2   );

    z2 = 200 * robustInv2 * 0.5 * error2;

    %     qr1err1 = 0.5*(qc(1) - q1(1))^2;
    %     qr2err1 = 0.5*(qc(1) - q2(1))^2;
    %     qerr1 = [qr1err1;  qr2err1];
    %     jacob_qerr1 = [-(qc(1) - q1(1)) zeros(1,13);];
    %     jacob_qerr1 = [jacob_qerr1; zeros(1,7)  -(qc(1) - q1(1)) zeros(1,6);];

    q_cc = [q_c';q_c'];
    J_s = (q - q_cc)';
    s = 0.5 * sum((q - q_cc).^2);
    lambda = 0.5;
    k = 0.5;
    dot_s = -lambda * s;
    Js_Jr = [J_s; jacob1];

    robustInvJs_Jr = Js_Jr' * pinv(Js_Jr * Js_Jr' + 0.001 * eye(9));
    robustInvJ_s = J_s' * pinv(J_s * J_s' + 0.001 * eye(1));
    P_J_s = eye(14) - robustInvJ_s * J_s;
    P_Js_Jr = eye(14) - robustInvJs_Jr * Js_Jr;

    z1 = gain * k * robustInv1 * error;
    z3 = pinv(J_s)*dot_s;

    % Prioritizing the tasks
    bv = boolFunc(dq, q, q_min, q_max);

    if bv == true
    %         disp('----true-----')
        dq = gain * k * robustInv1 *error + Pjacob1*z2 + Pjacobaug * z3;

    else
    %         disp('----false----')
        dq = gain * k * pinv(J_s) * dot_s + P_J_s * z1 + P_Js_Jr * z2;
    end


    %     dq = dq + Pjacobaug*( 100*robustInv2*error2   ); 



    % Outputs
    dq1 = dq(1:7);
    dq2 = dq(8:14);

    q1 = q1 + dq1*sampling_time;
    q2 = q2 + dq2*sampling_time;
    q = [q1;q2];

    dqad_upd = panda_bimanual.absolute_pose(q);
    p_curr = dqad_upd.translation.q(2:4);

    % Storing data for plotting
    data.fep_xd(:,jj) = task_dq.translation.q(2:4);
    data.fep_xm(:,jj) = panda_bimanual.absolute_pose(q).translation.q(2:4);
    data.fep_rel_error_norm(jj) = norm(error);    
    data.fep_abs_error_norm(jj) = norm(error2);
    data.fep_q_1(:,jj) = q1;
    data.fep_q_2(:,jj) = q2;
    data.fep_ee_pos_error_norm(:,jj) = norm(pos_err);
    data.cf_force(:,jj) = norm(force);
    data.min_svd(:,jj) = min_sv;

    if FLAGPLOT 

        plot(fep1, q1);
        plot(fep2, q2); 

        % Plot the obstacle sphere(s)
        [X,Y,Z] = sphere;
        for j=1:size(obs_pos,1)
            surf(X * obs_rad + obs_pos(j,1),Y * obs_rad + obs_pos(j,2), Z * obs_rad + obs_pos(j,3));
            hold on
        end
        axis equal;
        drawnow;   
    end

    if sum( q1 < q_min' ) > 0
        disp('Lower Joint Limit: Manip 1')
        double( q1 < q_min' )'*diag([1:7])
    end
    if sum( q1 > q_max' ) > 0
        disp('Upper Joint Limit: Manip 1')
        double( q1 > q_max' )'*diag([1:7])
    end
    if sum( q2 < q_min' ) > 0
        disp('Lower Joint Limit: Manip 2')
        double( q2 < q_min' )'*diag([1:7])
    end
    if sum( q2 > q_max' ) > 0
        disp('Upper Joint Limit: Manip 2')
        double( q2 > q_max' )'*diag([1:7])
    end

    jj = jj + 1;
    pos_err = p_goal_1 - p_curr;
end
toc

% For plotting
tt = linspace(0,jj * sampling_time,length(data.fep_abs_error_norm));

%% Plots

% Desired and actual translation 
figure(); 
p1 = plot(tt,data.fep_xd,'m--','LineWidth',3);
hold on, grid on 
p2 = plot(tt,data.fep_xm,'b','LineWidth',2);
legend([p1(1),p2(1)],'xdesired(translation)','xmeasured');

% Error plots
figure(); 
p3 = plot(tt,data.fep_abs_error_norm,'r','LineWidth',2);
hold on, grid on
p4 = plot(tt,data.fep_rel_error_norm,'g','LineWidth',2);
hold on, grid on
legend([p3(1), p4(1)],'Absolute Error norm',...
    'Relative Error norm'); 

figure(); 
p5 = plot(tt,data.fep_abs_error_norm,'r','LineWidth',3);
hold on, grid on
p6 = plot(tt,data.fep_ee_pos_error_norm,'g','LineWidth',2);
hold on, grid on
legend([p5(1), p6(1)],'Absolute Error norm',...
    'EE Position error norm');

% Variation of force over time along with minimum singular value
figure();
p7 = plot(tt,data.cf_force,'b', 'LineWidth',2);
hold on, grid on
p8 = plot(tt,data.min_svd,'g', 'LineWidth',2);
legend([p7(1), p8(1)],'Evolution of forces over time',...
    'Evolution of the minimum singluar value');
title("Evolution of the forces and minimum singular value over time");

% Visualize the trajectory in SE(3) with the obstacle
figure, plot3(data.fep_xm(1,:),data.fep_xm(2,:),data.fep_xm(3,:),'r','LineWidth',2);
title("Path towards goal for CF based obstacle avoidance");
hold on, grid on
% mesh(tx, ty, M);
[X,Y,Z] = sphere;
surf(X * obs_rad + obs_pos(1),Y * obs_rad + obs_pos(2), Z * obs_rad + obs_pos(3));
axis equal;
hold off

% Visualize the normal trajectory with just CF's (no robot)
figure();
[X,Y,Z] = sphere;
for j=1:size(obs_pos,1)
    surf(X * obs_rad + obs_pos(j,1),Y * obs_rad + obs_pos(j,2), Z * obs_rad + obs_pos(j,3));
    hold on
end
axis equal;
plot3(x, y, z, "r--",'LineWidth',3);
xlabel('x-axis')
ylabel('y-axis')
zlabel('z-axis')
title("Path towards goal");
hold on
plot3(data.fep_xm(1,:),data.fep_xm(2,:),data.fep_xm(3,:),'b','LineWidth',2);
hold off


% Check joint positions for Franka 1
figure();
annotation('textbox', [0 0.9 1 0.1], 'String', 'Check if joint positions stay in the centre for Franka1', 'EdgeColor', 'none','HorizontalAlignment', 'center')
for j=1:size(data.fep_q_1)
    subplot(4,2,j)
    plot(tt,data.fep_q_1(j,:),'g-',[tt(1) , tt(end)],[q_min(j),q_min(j)],'r-',[tt(1) , tt(end)],[q_max(j), q_max(j)],'r-')
    grid on
    ylabel(strcat('q_{',num2str(j),'}'))
end

% Check joint positions for Franka 2
figure();
annotation('textbox', [0 0.9 1 0.1], 'String', 'Check if joint positions stay in the centre for Franka2', 'EdgeColor', 'none','HorizontalAlignment', 'center')
for j=1:size(data.fep_q_2)
    subplot(4,2,j)
    plot(tt,data.fep_q_2(j,:),'g-',[tt(1) , tt(end)],[q_min(j),q_min(j)],'r-',[tt(1) , tt(end)],[q_max(j), q_max(j)],'r-')
    grid on
    ylabel(strcat('q_{',num2str(j),'}'))
end

%% Functions for Circular Fields

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function [x,xdot, xddot, y,ydot, yddot, z, zdot, zddot,t] = ...
    circularFields(robot_pos, robot_vel, goal, obs_pos, obs_rad, k_attr, k_circ)

% Parameters
ROBOT_MASS = 1.0;
time_step = 0.01;
V_MAX = 0.05;

% Distance to goal 
d = norm(goal - robot_pos);

i = 1;
% Control Loop
while d > 0.01

    % reseting the forces so that residuals are removed
    force = zeros(1,3);

    % Add the attractor forces
    att_force = AttractorForces(goal, robot_pos, k_attr);
    force = force + att_force;

    for j = 1:size(obs_pos,1)

      % Add circular forces for each obstacle
      circ_force = CurrForces(obs_pos(j,:), obs_rad, robot_pos, robot_vel, goal, k_circ);

      force = force + circ_force;

    end

    acc = force / ROBOT_MASS;

    robot_acc = acc;
    robot_vel = robot_vel + acc * time_step;
    vel_norm = norm(robot_vel);
    
    if vel_norm > V_MAX
       robot_vel = robot_vel * (V_MAX/vel_norm); 
    end
    
    robot_pos = robot_pos + 0.5 * acc * time_step * time_step+ robot_vel * time_step;
    
    % If the robot is too close to the goal slow it down
    if d < 0.1
        robot_vel = robot_vel * (10*d);
    end  
    
    % Update the distance to goal
    d = norm(goal - robot_pos);

    x(i) = robot_pos(1);
    xdot(i) = robot_vel(1);
    xddot(i) = robot_acc(1);
    y(i) = robot_pos(2);
    ydot(i) = robot_vel(2);
    yddot(i) = robot_acc(2);
    z(i) = robot_pos(3);
    zdot(i) = robot_vel(3);
    zddot(i) = robot_acc(3);
    t(i) = i * time_step;
    
    i = i + 1;
end

end


%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function att_force = AttractorForces(goal, robot_pos, k_attr)

% Function that computes the attractive forces

goal_vec = (goal - robot_pos);
goal_vec = goal_vec / norm(goal_vec);
att_force = k_attr * goal_vec;

end


%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function curr_force = CurrForces(obs_pos, obs_rad, robot_pos, robot_vel, goal, k_circ)

% Function that computes the circular forces

dist_obs = norm(robot_pos - obs_pos) - obs_rad;
curr_force = zeros(1,3);
if dist_obs < 4 * obs_rad
   if norm(robot_vel) ~= 0
       dist = comp_dist(robot_pos, obs_pos, goal);
       dist = dist/norm(dist);
       curr_force = (k_circ / dist_obs^2) * cross(robot_vel, cross(dist,robot_vel));
   end
end
end