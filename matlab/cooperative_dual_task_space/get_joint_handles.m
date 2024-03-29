function handles = get_joint_handles(vi,id)

robot1_name = 'Franka1';
robot2_name = 'Franka2';
handles = struct('id',id);

%% arm joints
armJoints1 = -ones(1,7);
for i=1:7
    [res,armJoints1(i)] = vi.vrep.simxGetObjectHandle(id,[robot1_name,'_joint',num2str(i)],vi.vrep.simx_opmode_oneshot_wait);

end
handles.armJoints1 = armJoints1;

armJoints2 = -ones(1,7);
for i=1:7
    [res,armJoints2(i)] = vi.vrep.simxGetObjectHandle(id,[robot2_name,'_joint',num2str(i)],vi.vrep.simx_opmode_oneshot_wait);

end
handles.armJoints2 = armJoints2;

%% streaming
for i=1:7
    vi.vrep.simxGetJointPosition(id,armJoints1(i),vi.vrep.simx_opmode_streaming);

    vi.vrep.simxGetObjectFloatParameter(id,armJoints1(i),2012,vi.vrep.simx_opmode_streaming);

end

for i=1:7
    vi.vrep.simxGetJointPosition(id,armJoints2(i),vi.vrep.simx_opmode_streaming);

    vi.vrep.simxGetObjectFloatParameter(id,armJoints2(i),2012,vi.vrep.simx_opmode_streaming);

end

% Make sure that all streaming data has reached the client at least once
vi.vrep.simxGetPingTime(id);
end
