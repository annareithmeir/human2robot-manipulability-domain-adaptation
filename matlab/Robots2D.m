% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbDOFs = 4;
% armLength = 1; 
% L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
% robot4 = SerialLink(repmat(L1,nbDOFs,1)); 
% q4 = [-90*pi/180 0 0 0 ]; % Initial robot configuration
% plot(robot4, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
% 
% npoints=20;
% delta=(90*pi/180+90*pi/180)/npoints;

%4DoF around joint1
% manips1=[];
% for i= -90*pi/190:delta:90*pi/180
%     q4 = [i 0 0 0 ]; % Initial robot configuration
%     plot(robot4, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
%     J_Me_d = robot4.jacob0(q4); % Current Jacobian
%     J_Me_d = J_Me_d(1:3,:);
%     Me_d = (J_Me_d*J_Me_d');
%     manips1=[manips1;reshape(Me_d,1,9)];
% end
% dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/4dof/link1/manipulabilities_20.csv", manips1, 'delimiter', ',', 'precision', 64);

%4DoF around joint2
% q4 = [-90*pi/180 0 0 0 ]; % Initial robot configuration
% plot(robot4, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
% 
% manips2=[];
% for i= -90*pi/190:delta:90*pi/180
%     q4 = [ 0 i 0 0 ]; % Initial robot configuration
%     plot(robot4, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
%     J_Me_d = robot4.jacob0(q4); % Current Jacobian
%     J_Me_d = J_Me_d(1:3,:);
%     Me_d = (J_Me_d*J_Me_d');
%     manips2=[manips2;reshape(Me_d,1,9)];
% end
% dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/4dof/link2/manipulabilities_20.csv", manips2, 'delimiter', ',', 'precision', 64);




% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all;
% nbDOFs = 2;
% armLength = 1; % For C
% L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
% robot = SerialLink(repmat(L1,nbDOFs,1)); 
% 
% npoints=20;
% delta=(90*pi/180+90*pi/180)/npoints;
% 
% %4DoF around joint1
% q4 = [-90*pi/180 0 ]; % Initial robot configuration
% plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
% manips3=[];
% for i= -90*pi/190:delta:90*pi/180
%     q4 = [i 0 ]; % Initial robot configuration
%     plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
%     J_Me_d = robot.jacob0(q4); % Current Jacobian
%     J_Me_d = J_Me_d(1:3,:);
%     Me_d = (J_Me_d*J_Me_d');
%     manips3=[manips3;reshape(Me_d,1,9)];
% end
% dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/2dof/link1/manipulabilities_20.csv", manips3, 'delimiter', ',', 'precision', 64);


%4DoF around joint2
% q4 = [ 0 -90*pi/180 ]; % Initial robot configuration
% plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
% 
% manips4=[];
% for i= -90*pi/190:delta:90*pi/180
%     q4 = [ 0 i]; % Initial robot configuration
%     plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
%     J_Me_d = robot.jacob0(q4); % Current Jacobian
%     J_Me_d = J_Me_d(1:3,:);
%     Me_d = (J_Me_d*J_Me_d');
%     manips4=[manips4;reshape(Me_d,1,9)];
% end
% dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/2dof/link2/manipulabilities_20.csv", manips4, 'delimiter', ',', 'precision', 64);


% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % scaled 2DoF
% nbDOFs = 2;
% armLength = 3; % For C
% L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
% robot = SerialLink(repmat(L1,nbDOFs,1)); 
% 
% npoints=20;
% delta=(90*pi/180+90*pi/180)/npoints;
% 
% %4DoF around joint1
% q4 = [-90*pi/180 0 ]; % Initial robot configuration
% plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
% manips3=[];
% for i= -90*pi/190:delta:90*pi/180
%     q4 = [i 0 ]; % Initial robot configuration
%     plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
%     J_Me_d = robot.jacob0(q4); % Current Jacobian
%     J_Me_d = J_Me_d(1:3,:);
%     Me_d = (J_Me_d*J_Me_d');
%     manips3=[manips3;reshape(Me_d,1,9)];
% end
% dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/2dof_scaled/link1/manipulabilities_20.csv", manips3, 'delimiter', ',', 'precision', 64);


%4DoF around joint2
% q4 = [ 0 -90*pi/180 ]; % Initial robot configuration
% plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
% 
% manips4=[];
% for i= -90*pi/190:delta:90*pi/180
%     q4 = [ 0 i]; % Initial robot configuration
%     plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
%     J_Me_d = robot.jacob0(q4); % Current Jacobian
%     J_Me_d = J_Me_d(1:3,:);
%     Me_d = (J_Me_d*J_Me_d');
%     manips4=[manips4;reshape(Me_d,1,9)];
% end
% dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/2dof_scaled/link2/manipulabilities_20.csv", manips4, 'delimiter', ',', 'precision', 64);

% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % vertical 2DoF
nbDOFs = 2;
armLength = 1; % For C
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
T=[0 0 -1 0;
   0 1 0 0;
  1 0 0 0;
   0 0 0 1];
robot = SerialLink(repmat(L1,nbDOFs,1), 'base', T); 

npoints=20;
delta=(90*pi/180+90*pi/180)/npoints;

%4DoF around joint1
% q4 = [-90*pi/180 0 ]; % Initial robot configuration
% plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
% manips3=[];
% for i= -90*pi/190:delta:90*pi/180
%     q4 = [i 0 ]; % Initial robot configuration
%     plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
%     J_Me_d = robot.jacob0(q4); % Current Jacobian
%     J_Me_d = J_Me_d(1:3,:);
%     Me_d = (J_Me_d*J_Me_d');
%     manips3=[manips3;reshape(Me_d,1,9)];
% end
% dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/2dof_vertical/link1/manipulabilities_20.csv", manips3, 'delimiter', ',', 'precision', 64);

% 
% %4DoF around joint2
% q4 = [ 0 -90*pi/180 ]; % Initial robot configuration
% plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
% 
% manips4=[];
% for i= -90*pi/190:delta:90*pi/180
%     q4 = [ 0 i]; % Initial robot configuration
%     plot(robot, q4, 'noshading','notiles','linkcolor','black','noshadow','noname');
%     J_Me_d = robot.jacob0(q4); % Current Jacobian
%     J_Me_d = J_Me_d(1:3,:);
%     Me_d = (J_Me_d*J_Me_d');
%     manips4=[manips4;reshape(Me_d,1,9)];
% end
% dlmwrite("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/2dof_vertical/link2/manipulabilities_20.csv", manips4, 'delimiter', ',', 'precision', 64);
