function dist = comp_dist(robot_pos, obs_pos, goal)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
line_normal = goal - robot_pos;
line_normal = line_normal/norm(line_normal);
robot_obs = robot_pos - obs_pos;
dist = robot_obs - line_normal * dot(robot_obs, line_normal);
if norm(dist) < 1e-10
    dist = -1 + 2 * rand(size(obs_pos));
end

