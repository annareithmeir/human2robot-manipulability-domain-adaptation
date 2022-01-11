% Control rHuman model towards q position

data_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/towards_singularities/human"

rhuman = rHuManModel('shoulderHeight',1.35,'verbose',true);
theta = [90; 30; 0; 40;   0; 0; 0]*pi/180;
% rhuman.plot(theta);
% view(-70,22);
% axis([-0.6 0.6 -0.3 0.6 0.0 1.8]);

position = rhuman.getPos(theta)
orientation = rhuman.getOrientation(theta)

e=0.05;
e_cnt = 9999;
q_desired = [90; 30; 0; 0;   0; 0; 0]*pi/180; 
x_desired = rhuman.getFKM(q_desired);
q_cnt=theta;

while (norm(e_cnt) > e)
    rhuman.plot(q_cnt);
    view(-70,22);
    axis([-0.6 0.6 -0.3 0.6 0.0 1.8]);
    
    x = rhuman.getFKM(q_cnt);
    e_cnt = vec8(x-x_desired);
%     J = rhuman.getJacobGeom(q_cnt);
    J = rhuman.kine.jacobian([q_cnt(1);  q_cnt(2);  q_cnt(1);  -q_cnt(3);    q_cnt(4); q_cnt(5); q_cnt(6); q_cnt(7);  ]);
    u = -1 * pinv(J) * e_cnt;
    q_cnt2 = q_cnt + u';
    norm(e_cnt)
end

