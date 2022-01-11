%% Script for dynamic control of FE Panda Arms in the Cooperative Dual Space

clear;
close all;
clc;

include_namespace_dq; 


fep1 = getFrankaKinematics();
fep2 = getFrankaKinematics();
baseR1 = DQ([0.887010404008924;0;0;0.461749437660648;-2.236195910568693e-08;-0.067005303402133;0.702186583093093;4.295682628495998e-08]);
baseR2 = DQ([0.499999888056226;0;0;-0.866025468415197;5.484533668703717e-08;-0.384077708534888;-0.650989969524217;3.166496044753432e-08]);

baseR1 = baseR1*inv(norm(baseR1));
fep1.set_reference_frame(baseR1);
fep1.set_base_frame(baseR1);
fep2.set_reference_frame(baseR2);
fep2.set_base_frame(baseR2);

%% Default Start
q1 = [2.722713708877564;-0.209439516067505;0.261799395084381;-1.570796370506287;0;2.792526721954346;1.570796370506287];
q2 = [2.722713708877564;-0.209439516067505;0.436332315206528;-1.570796370506287;0;2.792526721954346;1.570796370506287];
q = [q1;q2];

dq = zeros(14,1);
dq_init_1 = fep1.fkm(q1);
dq_init_2 = fep2.fkm(q2);

%% Load DQ Robotics kinematics
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

%% Field based Motion planning final condition and parameters

% q1_goal = [2.4384 -0.3518 0.5410 -1.9826 0.3599 3.1207 1.3489];
q1_goal = [1.5708 1.3265 1.4486 -1.5708 0 3.3161 1.5708 ];

dq_goal_1 = fep1.fkm(q1_goal);
dq_goal_o_1 = dq_goal_1.rotation;

dqrd = panda_bimanual.relative_pose(q);
dqad_ant = panda_bimanual.absolute_pose(q);

task_d1 = dqrd.q;
sampling_time = 0.010;

p_init_1 = dqad_ant.translation.q(2:4);
ini_o_1 = dqad_ant.P;
p_goal_1 = dq_goal_1.translation.q(2:4);

Obstacles2 = [-1.294 0.7611 0.7509]; 
Obstacles3 = [-1.359 0.857 0.4839]; 
Obstacles4 = [-1.109 0.7071 0.4839]; 
obs_pos = [Obstacles2; Obstacles3; Obstacles4];
obs_rad = 0.1;

k_circ = 0.1;
k_attr = 0.2;
robot_pos = p_init_1';
robot_vel = zeros(1,3);


[x,xdot, xddot, y,ydot, yddot, z, zdot, zddot,t] = ...
    circularFields(robot_pos, robot_vel, p_goal_1', obs_pos, obs_rad, k_attr, k_circ);

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
    view(-76,18)
    drawnow;
pause    
end

%% Defining Constraints based on Primitives

% Current Orientation
r = fep1.fkm(q(1:7)).P;

% Desired Plücker line in k_
l = ini_o_1 * j_ * ini_o_1';

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

%% Two-arm control loop
pos_err = p_goal_1 - p_init_1;
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
    lz = r * j_ * r';
    
    % Line-static-Line angle Jacobian
    J_rz = haminus4(j_ * r')* Jr_r + hamiplus4(r*j_)*dqrd.C4 * Jr_r;
    
    % Error
    err_t = vec4( l - lz);
    e_n = norm(err_t)^2;
    err_max = 10;
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

    z1 = 1 * robustInv1 * error;
    z2 = -1 * robustJ_lerr * e_n;
    z3 = 100 * robustInv2 *error2;
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


% Visualize the normal trajectory with just CF's (no robot)
f = figure();
f.Renderer = 'painters';
[X,Y,Z] = sphere;
for j=1:size(obs_pos,1)
    surf(X * (obs_rad-0.03) + obs_pos(j,1),Y * (obs_rad-0.03) + obs_pos(j,2), Z * (obs_rad-0.03) + obs_pos(j,3));
    hold on
end
axis equal;
axis([-2.0 -0.5 0 1.5 0.0 1.1]);
plot3(x, y, z, 'r--', 'LineWidth',1);
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
set(gca,'zticklabel',[])
hold on
plot3(data.fep_xm(1,:),data.fep_xm(2,:),data.fep_xm(3,:),'b','LineWidth',1);
hold on

for j= 0:9
    i = 1 + j*200;
    if i == (9*200 + 1)
        i = size(t, 2);
    end
    plot(fep1, data.fep_q_1(:,i));
    plot(fep2, data.fep_q_2(:,i)); 
    hold on

    ori_r = [-1.261, 0.4374, 0.8966];
    cube_r_x = [-1.186, 0.3836, 0.889];
    cube_r_y = [-1.27, 0.438, 0.8041];
    cube_r_z = [-1.207, 0.5131, 0.8918];
    rx = 0.3*(cube_r_x - ori_r);
    ry = 0.07*(cube_r_y - ori_r);
    rz = (cube_r_z - ori_r);

    first = [data.fep_xm(1,1),data.fep_xm(2,1),data.fep_xm(3,1)];
    current = [data.fep_xm(1,i),data.fep_xm(2,i),data.fep_xm(3,i)];
    offset = current - first;

    pt1 = cube_r_z + 0.6*rz + rx - ry + offset;
    pt2 = cube_r_z + 0.6*rz + rx + ry + offset;
    pt3 = cube_r_z + 0.6*rz - rx + ry + offset;
    pt4 = cube_r_z + 0.6*rz - rx - ry + offset;
    pt5 = pt1 + 1.2*rz;
    pt6 = pt2 + 1.2*rz;
    pt7 = pt3 + 1.2*rz;
    pt8 = pt4 + 1.2*rz;

    patch([pt1(1),pt2(1),pt3(1),pt4(1)],...
          [pt1(2),pt2(2),pt3(2),pt4(2)],...
          [pt1(3),pt2(3),pt3(3),pt4(3)],'magenta');
    patch([pt5(1),pt6(1),pt7(1),pt8(1)],...
          [pt5(2),pt6(2),pt7(2),pt8(2)],...
          [pt5(3),pt6(3),pt7(3),pt8(3)],'blue');
    patch([pt1(1),pt4(1),pt8(1),pt5(1)],...
          [pt1(2),pt4(2),pt8(2),pt5(2)],...
          [pt1(3),pt4(3),pt8(3),pt5(3)],'yellow');
    patch([pt2(1),pt3(1),pt7(1),pt6(1)],...
          [pt2(2),pt3(2),pt7(2),pt6(2)],...
          [pt2(3),pt3(3),pt7(3),pt6(3)],'green');
    patch([pt1(1),pt2(1),pt6(1),pt5(1)],...
          [pt1(2),pt2(2),pt6(2),pt5(2)],...
          [pt1(3),pt2(3),pt6(3),pt5(3)],'red');
    patch([pt3(1),pt4(1),pt8(1),pt7(1)],...
          [pt3(2),pt4(2),pt8(2),pt7(2)],...
          [pt3(3),pt4(3),pt8(3),pt7(3)],'cyan');

    view(3)
    view(-52,23)
end


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