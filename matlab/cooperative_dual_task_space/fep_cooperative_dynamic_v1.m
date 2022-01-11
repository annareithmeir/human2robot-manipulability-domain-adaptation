% Script for dynamic control of FE Panda Arms in the Cooperative Dual Space

clear;
close all;
clc;

include_namespace_dq; 

%% Initialize V-REP interface
vi = DQ_VrepInterface;
vi.disconnect_all(); 
vi.connect('127.0.0.1',19997);
% vi.start_simulation();
% sim=remApi('remoteApi'); % using the prototype file (remoteApiProto.m)
% sim.simxFinish(-1);
% clientID=sim.simxStart('127.0.0.1',19997,true,true,5000,5);

% vi = DQ_VrepInterface;
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

%% Get Joint Handles

handles = get_joint_handles(vi,vi.clientID);
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
dq = [-0.03,-0.03,0,0,0.0,0,0]';
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
q = [q1;q2];

%% Task definitions

dqrd = panda_bimanual.relative_pose(q);

dqad_ant = panda_bimanual.absolute_pose(q);
dqad = DQ([cos(pi/32);0;sin(pi/32);0]) .* dqad_ant;

task_d = [dqad.q;dqrd.q];


% Sampling time 
sampling_time = 0.010; % V-REP's sampling time 

% Setting simulation to synchronous mode
vi.vrep.simxSynchronous(vi.clientID,true);   
vi.vrep.simxSynchronousTrigger(vi.clientID);
vi.vrep.simxSetFloatingParameter(vi.clientID, vi.vrep.sim_floatparam_simulation_time_step,...
    sampling_time, vi.vrep.simx_opmode_blocking);

vi.vrep.simxStartSimulation(vi.clientID, vi.vrep.simx_opmode_blocking);


% for j=1:7
%     vi.vrep.simxSetJointForce(vi.clientID,joint_handles1(j),abs(9999),vi.vrep.simx_opmode_blocking);
%     vi.vrep.simxSetJointTargetVelocity(vi.clientID,joint_handles1(j),40*0.4,vi.vrep.simx_opmode_blocking);
%     %---------------------------------
% end
% 
% for j=1:7
%     vi.vrep.simxSetJointForce(vi.clientID,joint_handles2(j),abs(9999),vi.vrep.simx_opmode_blocking);
%     vi.vrep.simxSetJointTargetVelocity(vi.clientID,joint_handles2(j),400*0.4,vi.vrep.simx_opmode_blocking);
%     %---------------------------------
% end
% i=1
% while i<=10
%     
% 
% 
%     for j=1:7
% 
%         % Changed to blocking mode
%         %---------------------------------
%         %             sim.simxSetJointTargetVelocity(clientID,joint_handles(j),set_vel,sim.simx_opmode_oneshot);            
%         vi.vrep.simxSetJointTargetVelocity(vi.clientID,joint_handles1(j),10*dq(j),vi.vrep.simx_opmode_blocking);            
%         vi.vrep.simxSetJointForce(vi.clientID,joint_handles1(j),abs(9999),vi.vrep.simx_opmode_blocking);
% %         sim.simxSetJointForce(clientID,joint_handles(j),abs(tau(j)),sim.simx_opmode_blocking);
%         %---------------------------------
% 
%     %             sim.simxSetJointForce(clientID,joint_handles(j),0.01,sim.simx_opmode_oneshot);
%     end
% % In synchronous mode this is not necessary        
% %         sim.simxPauseCommunication(clientID, 0);
% % Move vrep simulation one step up (pause to allow vrep
% % computations to work properly (just a safe measure, perhaps we do
% % need it)
% %---------------------------------
% vi.vrep.simxSynchronousTrigger(vi.clientID);
% pause(0.002)
% %---------------------------------
% i = 1 +1;       
% end

% %% Two-arm control
% 
epsilon = 0.001; %for the stop condition
error = epsilon+1;
i = 0;
iter = 1;
jj=1;
% 
% %The sweep motion (back and forth) will be performed twice
while jj <= 50   
%     disp('Inside loop')
%     for j=1:7
%         vi.vrep.simxSetJointForce(vi.clientID,joint_handles1(j),abs(9999),vi.vrep.simx_opmode_blocking);
%         vi.vrep.simxSetJointTargetVelocity(vi.clientID,joint_handles1(j),10*dq(j),vi.vrep.simx_opmode_blocking);
%         %---------------------------------
%     end
%     for j=1:7
%         vi.vrep.simxSetJointForce(vi.clientID,joint_handles2(j),abs(9999),vi.vrep.simx_opmode_blocking);
%         vi.vrep.simxSetJointTargetVelocity(vi.clientID,joint_handles2(j),10*dq(j),vi.vrep.simx_opmode_blocking);
%         %---------------------------------
%     end
%     %---------------------------------
%     vi.vrep.simxSynchronousTrigger(vi.clientID);
%     pause(0.002)
    %---------------------------------
%     
% %     for j=1:7
% %         vi.vrep.simxSetJointForce(vi.clientID,joint_handles1(j),abs(9999),vi.vrep.simx_opmode_blocking);
% %         vi.vrep.simxSetJointTargetVelocity(vi.clientID,joint_handles1(j),20*0.4,vi.vrep.simx_opmode_blocking);
% %         %---------------------------------
% %     end
% %     %standard control law

q1 = fep_vreprobot1.get_q_from_vrep();
q2 = fep_vreprobot2.get_q_from_vrep();
q = [q1;q2];

    nerror_ant = error;
    jacob = [panda_bimanual.absolute_pose_jacobian(q);...
             panda_bimanual.relative_pose_jacobian(q)];
    taskm =  [vec8(panda_bimanual.absolute_pose(q));...
              vec8(panda_bimanual.relative_pose(q))];
          
    N1 = haminus8(dqad)*panda_bimanual.absolute_pose(q).C8*panda_bimanual.absolute_pose_jacobian(q);
    N2 = haminus8(dqrd)*panda_bimanual.relative_pose(q).C8*panda_bimanual.relative_pose_jacobian(q);
    
    N = [N1;N2];
    robustInv = N'*pinv(N*N' + 0.001*eye(16));

    error = task_d-taskm;
%     dq = 20*pinv(jacob)*0.5*error;
    dq = 20 * robustInv * 0.5 * error;
    dq1 = dq(1:7);
    dq2 = dq(8:14);
% %     
    q = q + sampling_time*dq; 
% % %     
% % %     % Send desired values to the robots
% % %     fep_vreprobot1.send_q_to_vrep(q(1:7)');
% % %     fep_vreprobot2.send_q_to_vrep(q(8:14)');
% %     
    for j=1:7
        vi.vrep.simxSetJointForce(vi.clientID,joint_handles1(j),abs(9999),vi.vrep.simx_opmode_blocking);
        vi.vrep.simxSetJointTargetVelocity(vi.clientID,joint_handles1(j), dq1(j),vi.vrep.simx_opmode_blocking);
        %---------------------------------
    end
    
    for j=1:7
        vi.vrep.simxSetJointForce(vi.clientID,joint_handles2(j),abs(9999),vi.vrep.simx_opmode_blocking);
        vi.vrep.simxSetJointTargetVelocity(vi.clientID,joint_handles2(j), dq2(j),vi.vrep.simx_opmode_blocking);
        %---------------------------------
    end
    
    
%     error'
%     norm(error)
    
%     In synchronous mode this is not necessary        
%         sim.simxPauseCommunication(clientID, 0);
%     Move vrep simulation one step up (pause to allow vrep
%     computations to work properly (just a safe measure, perhaps we do
%     need it)
%     ---------------------------------
    vi.vrep.simxSynchronousTrigger(vi.clientID);
    pause(0.002);
%     ---------------------------------
%     
% %     disp('After sending velocity')
% %     % Visualisation
% % 
% %     % Plot the arms
% % %     plot(kuka1,q(1:7)');
% % %     plot(kuka2,q(8:14)');
% % % 
% % %     % Plot the broom
% % %     aux = translation(two_arms.pose1(q))-translation(two_arms.pose2(q));
% % %     t_broom = aux*(1/norm(aux.q))*d_broom;
% % % 
% % %     broom_base=translation(two_arms.pose2(q));
% % %     broom_tip = broom_base+t_broom;
% % % 
% % %     set(line_handle1, 'Xdata', [broom_base.q(2), broom_tip.q(2)], 'Ydata', ...
% % %         [broom_base.q(3), broom_tip.q(3)], 'Zdata',...
% % %         [broom_base.q(4),broom_tip.q(4)]);
% % 
%     Verify if the sweep direction can be changed
    if(norm(nerror_ant - error) < epsilon)
        disp('changed')
        %Change the task
        temp = dqad;
        dqad = dqad_ant;
        dqad_ant = temp;

        task_d(1:8,1)=dqad.q;
        jj = jj+1;
    end
% %     drawnow;
    jj = jj + 1;
end

%% End V-REP
vi.stop_simulation();
vi.disconnect();