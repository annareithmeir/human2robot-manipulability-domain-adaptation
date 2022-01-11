function closeGripper(clientID, vi, object_handle, effector_handle)
%   CLOSEGRIPPER
%
%   Closes the gripper.
%
%   INPUTS:         CLIENTID -> Handle for V-rep client;
%                       VREP -> Handle for V-rep connection;
%              OBJECT_HANDLE -> Handle for the end-effectors;
%            EFFECTOR_HANDLE -> Handle for the end-effectors.
%
%   NO OUTPUTS
%

rtn=vi.vrep.simxSetObjectParent(clientID,object_handle,effector_handle.attach_point,true,vrep.simx_opmode_streaming);
vi.vrep.simxSetUIButtonProperty(clientID,effector_handle.gripper,20,vrep.sim_buttonproperty_isdown,vrep.simx_opmode_oneshot_wait);

end