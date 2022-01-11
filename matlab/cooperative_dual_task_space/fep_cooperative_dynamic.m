% Script for dynamic control of FE Panda Arms in the Cooperative Dual
% Space with Circular Fields based obstacle avoidance

clear;
close all;
clc;

include_namespace_dq; 

%% Initialize V-REP interface
vi = DQ_VrepInterface;
vi.disconnect_all(); 
vi.connect('127.0.0.1',19997);

%% Initialize VREP Robots
fep_vreprobot1 = FEpVrepRobot1('Franka1',vi);
fep_vreprobot2 = FEpVrepRobot2('Franka2',vi);

%% Load DQ Robotics kinematics
fep1  = fep_vreprobot1.kinematics();
fep2  = fep_vreprobot2.kinematics();

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
dq = zeros(14,1);

%% Get Joint Handles

handles = get_joint_handles(vi, vi.clientID);
joint_handles1 = handles.armJoints1;
joint_handles2 = handles.armJoints2;

% get initial state of the robot (Using VRep Matlab Remote API directly)
qstr = '[ ';
qdotstr = '[ ';

for j=1:7
    [res,q(j)] = vi.vrep.simxGetJointPosition(vi.clientID,joint_handles1(j),vi.vrep.simx_opmode_buffer);
    [res,qdot(j)] = vi.vrep.simxGetObjectFloatParameter(vi.clientID,joint_handles1(j),2012,vi.vrep.simx_opmode_buffer);
    qstr = [qstr,num2str(q(j)),' '];
    qdotstr = [qdotstr,num2str(qdot(j)),' '];
end

qstr = [qstr,']'];
qdotstr = [qdotstr,']'];
disp('Initial Joint positions for Franka1: ');
disp(qstr);
disp('Initial Joint Velocities for Franka1: ');
disp(qdotstr);
qstr = '[ ';
qdotstr = '[ ';

for j=1:7
    [res,q(j)] = vi.vrep.simxGetJointPosition(vi.clientID,joint_handles2(j),vi.vrep.simx_opmode_buffer);
    [res,qdot(j)] = vi.vrep.simxGetObjectFloatParameter(vi.clientID,joint_handles2(j),2012,vi.vrep.simx_opmode_buffer);
    qstr = [qstr,num2str(q(j)),' '];
    qdotstr = [qdotstr,num2str(qdot(j)),' '];
end

qstr = [qstr,']'];
qdotstr = [qdotstr,']'];
disp('Initial Joint positions for Franka2: ');
disp(qstr);
disp('Initial Joint Velocities for Franka2: ');
disp(qdotstr);


% get initial state of the robots using DQ functions
q1 = fep_vreprobot1.get_q_from_vrep();
q2 = fep_vreprobot2.get_q_from_vrep();

dq_init_1 = fep1.fkm(q1);
dq_init_2 = fep2.fkm(q2);

p_init_1 = dq_init_1.translation.q(2:4);
p_init_2 = dq_init_2.translation.q(2:4);

ini_o_1 = dq_init_1.rotation;
ini_o_2 = dq_init_2.rotation;

robot_pos = p_init_1';

q2 = fep_vreprobot2.get_q_from_vrep();
dq_init_2 = fep2.fkm(q2);

q = [q1;q2];

%% Field based Motion planning final condition and parameters

% q1_goal = [2.4384 -0.3518 0.5410 -1.9826 0.3599 3.1207 1.3489];
% q1_goal = [ 1.5708, -1.5708, -1.5708, -0.61087, 0, 2.1817, -1.5708 ];
% dq_goal_1 = fep1.fkm(q1_goal);
% p_goal_1 = dq_goal_1.translation.q(2:4);

dqrd = panda_bimanual.relative_pose(q);
dqad_ant = panda_bimanual.absolute_pose(q);

task_d1 = dqrd.q;
sampling_time = 0.010;

p_init_1 = dqad_ant.translation.q(2:4);
ini_o_1 = dqad_ant.P;

obs_rad = 0.06;

obs_pos = [-0.45   -0.01    0.62;...
           -0.45   +0.01    0.62;...
           -0.44   -0.01    0.61;...
           -0.44   +0.01    0.61;...
           -0.43   -0.01    0.62;...
           -0.43   +0.01    0.62;...
           -0.41   -0.01    0.62;...
           -0.41   +0.01    0.62;...
           -0.38   -0.01    0.622;...
           -0.38   +0.01    0.622;...
           -0.35   -0.01    0.62;...
           -0.35   +0.01    0.62;...
           -0.40   +0.00    0.375;...
           -0.30   +0.00    0.365];

% for s=0:5
%     for r=0:4
%         if (r ~= 4 || s ~= 2) && (r ~= 4 || s ~= 3)
%             obs_pos(end+1, :) = [-0.30, -0.1938 + (0.4/5)*s, 0.3 + 0.1*r];
%         end
%     end
% end
% 
% for s=0:5
%     for r=0:4
%         if (r ~= 4 || s ~= 2) && (r ~= 4 || s ~= 3)
%             obs_pos(end+1, :) = [+0.15, -0.1938 + (0.4/5)*s, 1.4 - 0.1*r];
%         end
%     end
% end

k_circ = 0.5;
k_attr = 0.2;
robot_pos = p_init_1';
% goal_pos = p_goal_1';
% goal_pos = [0.70, 0.0062, 0.53];
goal_pos = [-0.4   -0.0    0.45];
p_goal_1 = goal_pos';
robot_vel = [0.05, 0, 0];


[x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, t] = ...
    circularFields(robot_pos, robot_vel, goal_pos, obs_pos, obs_rad, k_attr, k_circ);


%% Visualize path
f = figure();
f.Renderer = 'painters';
[X,Y,Z] = sphere;
for j=1:size(obs_pos,1)
    surf(X * (obs_rad-0.03) + obs_pos(j,1),Y * (obs_rad-0.03) + obs_pos(j,2), Z * (obs_rad-0.03) + obs_pos(j,3));
    hold on
end
plot3(x, y, z, 'r--', 'LineWidth',1);
hold on

%% Task definitions

% dqad = DQ([cos(pi/32);0;sin(pi/32);0]) .* dqad_ant;
% dqad = ( DQ(1) + DQ.E*0.5*DQ([0 0 0 -0.15])*DQ(1) ) .* dqad_ant;  

%% Defining Constraints based on Primitives

% Current Orientation
r = fep1.fkm(q(1:7)).P;

% Desired Plücker line in k_
l = ini_o_1 * i_ * ini_o_1';


%% Data for analyzing later
data.fep_xd = [];
data.fep_xm = [];
data.fep_q_1 = [];
data.fep_q_2 = [];
data.norm_dq = [];
data.fep_abs_error_norm = [];
data.fep_rel_error_norm = [];
data.fep_ee_pos_error_norm = [];
data.deviation_angle = [];

% Setting simulation to synchronous mode
vi.vrep.simxSynchronous(vi.clientID,true);   
vi.vrep.simxSynchronousTrigger(vi.clientID);
vi.vrep.simxSetFloatingParameter(vi.clientID, vi.vrep.sim_floatparam_simulation_time_step,...
    sampling_time, vi.vrep.simx_opmode_blocking);

%% Start simulation
vi.vrep.simxStartSimulation(vi.clientID, vi.vrep.simx_opmode_blocking);

%% Two-arm control loop
pos_err = zeros(3,1);%p_goal_1 - p_init_1;
jj=1;
tic

disp('hey there')
while jj <= size(t,2)
    if ~mod(jj, 500)
        disp(jj)
    end
    
    q = [q1;q2];

    % Desired Trajectory (Taking points one by one from the CF)
    p = x(jj)*i_ + y(jj)*j_ + z(jj)*k_;
    task_dq = ini_o_1 + DQ.E * 0.5 * p * ini_o_1;
    task_d_abs = vec8(task_dq); 

    % Getting current pose and Jacobian
    jacob1 = panda_bimanual.relative_pose_jacobian(q);
    taskm1 = vec8(panda_bimanual.relative_pose(q));
    xm = panda_bimanual.absolute_pose(q);
    taskm2 = vec8(xm);  
    
    % Getting the geometric Jac
    geomJac = geomJ(panda_bimanual.absolute_pose_jacobian(q), panda_bimanual.absolute_pose(q));
    
    % Translation Jacobian for the absolute pose
    J_t = geomJac(4:6,:);    
    jacob3 = J_t;
    % Getting current error
    error  = task_d1-taskm1;
    error2 = task_dq.translation.q(2:4) - xm.translation.q(2:4);

    % PseudoInverse
    robustInv1 = jacob1'*pinv(jacob1*jacob1' + 0.001*eye(8) );
    robustInv2 = J_t'*pinv(J_t*J_t' + 0.001*eye(3) );    
    
    % Rotation Jacobian
    jacob2rot = panda_bimanual.absolute_pose_jacobian(q);
    Jr_r = jacob2rot(1:4,:);
    
    % Current Orientation
    r = panda_bimanual.absolute_pose(q).P;
    
    % Current Plücker line in k_
    lz = r * i_ * r';
    
    % Line-static-Line angle Jacobian
    J_rz = haminus4(i_ * r')* Jr_r + hamiplus4(r*i_)*dqrd.C4 * Jr_r;
    
    % Error
    err_t = vec4( l - lz);
    e_n = norm(err_t)^2;
    err_max = 0.01;
    err_min = 0;
    
    % Line Jacobian error def
    J_lerr = -2 * err_t' * J_rz;
    der_err_line = J_lerr*dq;
    robustJ_lerr = J_lerr' * pinv(J_lerr * J_lerr' + 0.001 * eye(1));
    jacob2 = J_lerr;
    
    % Calculating the angle between the plücker lines
    tmp1 = vec4(l);
    tmp2 = vec4(lz);
    angle = (180/pi) * acos(dot(tmp1, tmp2));
    
    % Set-based switching framework for tilt of the EE
    bvtilt = boolFunctilt(der_err_line, abs(angle) , err_min, err_max );

    % Joint limits
    q_cc = [q_c';q_c'];
    J_s = (q - q_cc)';
    jacob4 = J_s;
    s = 0.5 * sum((q - q_cc).^2);
    lambda = 0.5;
    k = 0.5;
    dot_s = -lambda * s;
    robustInvJ_s = J_s' * pinv(J_s * J_s' + 0.001 * eye(1));

    % Prioritizing the tasks
    bv = boolFunc(dq, q, [q_min;q_min], [q_max;q_max]);
    
    % Task1 = Relative Pose
    % Task2 = Tilt Jacobian
    % Task3 = Absolute Pose
    % Task4 = Joint Limits
      
    if norm(error2) < 0.05
        g_z3 = norm(error2)/0.05;
    else
        g_z3 = 1;
    end
    
    z1 = 1 * robustInv1 * error;
    z2 = -1 * robustJ_lerr * e_n;
    z3 = g_z3 * 100 * robustInv2 *error2;
    z4 = 1 * robustInvJ_s * dot_s;
    
    if     bvtilt == true && bv == true
%         TaskOrder = [jacob1;jacob3;jacob2;jacob4];
        Pjacobaug3 = eye(14) - pinv(jacob1)*jacob1;
        Pjacobaug2 = eye(14) - pinv([jacob1; jacob3])*[jacob1; jacob3];
        Pjacobaug4 = eye(14) - pinv([jacob1; jacob3; jacob2])*[jacob1; jacob3; jacob2];
        dq1 = z1;
        dq2 = Pjacobaug3 * z3;
        dq3 = Pjacobaug2 * z2;
        dq4 = Pjacobaug4 * z4;
        dq =  dq1 + dq2 + dq3 + dq4;
        
    elseif bvtilt == true && bv == false
%         TaskOrder = [jacob1;jacob4;jacob3;jacob2];
        Pjacobaug4 = eye(14) - pinv(jacob1)*jacob1;
        Pjacobaug3 = eye(14) - pinv([jacob1; jacob4])*[jacob1; jacob4];
        Pjacobaug2 = eye(14) - pinv([jacob1; jacob4; jacob3])*[jacob1; jacob4; jacob3];
        dq1 = z1;
        dq2 = Pjacobaug4 * z4;
        dq3 = Pjacobaug3 * z3;
        dq4 = Pjacobaug2 * z2;
        dq =  dq1 + dq2 + dq3 + dq4;
        
    elseif bvtilt == false && bv == true
%         TaskOrder = [jacob1;jacob2;jacob3;jacob4];
        Pjacobaug2 = eye(14) - pinv(jacob1)*jacob1;
        Pjacobaug3 = eye(14) - pinv([jacob1; jacob2])*[jacob1; jacob2];
        Pjacobaug4 = eye(14) - pinv([jacob1; jacob2; jacob3])*[jacob1; jacob2; jacob3];
        dq1 = z1;
        dq2 = Pjacobaug2 * z2;
        dq3 = Pjacobaug3 * z3;
        dq4 = Pjacobaug4 * z4;
        dq =  dq1 + dq2 + dq3 + dq4;
        
    elseif bvtilt == false && bv == false
%         TaskOrder = [jacob1;jacob2;jacob4;jacob3];
        Pjacobaug2 = eye(14) - pinv(jacob1)*jacob1;
        Pjacobaug4 = eye(14) - pinv([jacob1; jacob2])*[jacob1; jacob2];
        Pjacobaug3 = eye(14) - pinv([jacob1; jacob2; jacob4])*[jacob1; jacob2; jacob4];
        dq1 = z1;
        dq2 = Pjacobaug2 * z2;
        dq3 = Pjacobaug4 * z4;
        dq4 = Pjacobaug3 * z3;
        dq =  dq1 + dq2 + dq3 + dq4;
    end

    % Outputs
    dq1 = dq(1:7);
    dq2 = dq(8:14);
    


    q1 = q1 + dq1*sampling_time;
    q2 = q2 + dq2*sampling_time;

    % Restricting Joint Limits
%     if bv == false
%         q_max2 = q_max;
%         q_max2(6)= 3.5;
%         q1 = min(q_max2', q1);
%         q1 = max(q_min', q1);
%         q2 = min(q_max2', q2);
%         q2 = max(q_min', q2);        
%     end    
    
    q = [q1;q2];
    
    dqad_upd = panda_bimanual.absolute_pose(q);
    p_curr = dqad_upd.translation.q(2:4);
    
    % Send desired values to the robot; needed for v-rep
    fep_vreprobot1.send_q_to_vrep(q1);
    fep_vreprobot2.send_q_to_vrep(q2);
    vi.vrep.simxSynchronousTrigger(vi.clientID);
    pause(0.002);

    
    % Storing data for plotting
    data.fep_xd(:,jj) = task_dq.translation.q(2:4);
    data.fep_xm(:,jj) = panda_bimanual.absolute_pose(q).translation.q(2:4);
    data.fep_rel_error_norm(jj) = norm(error);    
    data.fep_abs_error_norm(jj) = norm(error2);
    data.fep_q_1(:,jj) = q1;
    data.fep_q_2(:,jj) = q2;
    data.norm_dq(:,jj) = dq;
    data.deviation_angle(jj) = angle; 
    data.fep_ee_pos_error_norm(:,jj) = norm(pos_err);


%     if sum( q1 < q_min' ) > 0
%         disp('Lower Joint Limit: Manip 1')
%         double( q1 < q_min' )'*diag([1:7])
%     end
%     if sum( q1 > q_max' ) > 0
%         disp('Upper Joint Limit: Manip 1')
%         double( q1 > q_max' )'*diag([1:7])
%     end
%     if sum( q2 < q_min' ) > 0
%         disp('Lower Joint Limit: Manip 2')
%         double( q2 < q_min' )'*diag([1:7])
%     end
%     if sum( q2 > q_max' ) > 0
%         disp('Upper Joint Limit: Manip 2')
%         double( q2 > q_max' )'*diag([1:7])
%     end

    jj = jj + 1;
    pos_err = p_goal_1 - p_curr;

end
toc

% For plotting
tt = linspace(0,jj*sampling_time,length(data.fep_abs_error_norm));

%% End V-REP
vi.stop_simulation();
vi.disconnect();

%% Plots

% Desired and actual translation 
figure(); 
p1 = plot(tt,data.fep_xd,'m--','LineWidth',3);
hold on, grid on 
p2 = plot(tt,data.fep_xm,'b','LineWidth',2);
axis tight;
legend([p1(1),p2(1)],'xdesired(translation)','xmeasured');


% Error plots
figure(); 
p3 = plot(tt,data.fep_abs_error_norm,'r','LineWidth',2);
hold on, grid on
p4 = plot(tt,data.fep_rel_error_norm,'g','LineWidth',2);
hold on, grid on
axis tight;
legend([p3(1), p4(1)],'Absolute Error norm',...
    'Relative Error norm'); 


% normed joint velocities
figure(); 
plot(tt,data.norm_dq,'r','LineWidth',2);
hold on, grid on
axis tight;


% deviation angle
figure();
plot(tt, data.deviation_angle,'r','LineWidth',2);
hold on, grid on
title("Deviation Angle plot");
axis tight;


% Check joint positions for Franka 1
figure();
annotation('textbox', [0 0.9 1 0.1], 'String', 'Check if joint positions stay in the centre for Franka1', 'EdgeColor', 'none','HorizontalAlignment', 'center')
for j=1:size(data.fep_q_1)
    subplot(4,2,j)
    plot(tt,data.fep_q_1(j,:),'g-',[tt(1) , tt(end)],[q_min(j),q_min(j)],'r-',[tt(1) , tt(end)],[q_max(j), q_max(j)],'r-')
    grid on
    axis tight;
    ylabel(strcat('q_{',num2str(j),'}'))
end


% Check joint positions for Franka 2
figure();
annotation('textbox', [0 0.9 1 0.1], 'String', 'Check if joint positions stay in the centre for Franka2', 'EdgeColor', 'none','HorizontalAlignment', 'center')
for j=1:size(data.fep_q_2)
    subplot(4,2,j)
    plot(tt,data.fep_q_2(j,:),'g-',[tt(1) , tt(end)],[q_min(j),q_min(j)],'r-',[tt(1) , tt(end)],[q_max(j), q_max(j)],'r-')
    grid on
    axis tight;
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

    % Add circular forces for each obstacle
    for j = 1:size(obs_pos,1)
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
    if d < 0.01
        robot_vel = robot_vel * (99*d);
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
dist_obs = norm(robot_pos - obs_pos) - 1.01*obs_rad;
curr_force = zeros(1,3);
if dist_obs < 3 * obs_rad
   if norm(robot_vel) ~= 0
       dist = comp_dist(robot_pos, obs_pos, goal);
       dist = dist/norm(dist);
       curr_force = (k_circ / dist_obs^2) * cross(robot_vel, cross(dist, robot_vel));
   end
end
end