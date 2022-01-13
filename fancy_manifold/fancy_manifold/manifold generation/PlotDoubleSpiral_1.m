R = 0.2;
Y = 0.1;
t = 0:0.04*pi:4*pi
x1 = R*cos(t);
y1 = R*sin(t);
z1 = Y*t;

r = 0.01;
theta = 0:0.1*pi:2*pi;
ring_x = r*cos(theta');
ring_y = r*sin(theta');

basic_ring = [ring_x,ring_y,zeros(length(theta),1)];

Points_A = [];

for i = 1:length(t)
    
    alpha_x = -R*sin(t(i))/sqrt(R^2+Y^2);
    alpha_y = R*cos(t(i))/sqrt(R^2+Y^2);
    alpha_z = Y/sqrt(R^2+Y^2);
    
    A = [alpha_x, alpha_y, alpha_z];
    B = [0,0,1];
    v = cross(A,B);
    s = sqrt(v(1)^2+v(2)^2+v(3)^2);
    v_x = [0, -v(3),v(2);
           v(3),0,-v(1);
           -v(2),v(1),0];
    
    c = A*B';
    RM = eye(3)+v_x + v_x*v_x*(1-c)/s^2;

%      Rx = [1,0,0;
%       0,cos(alpha_x),-sin(alpha_x);
%       0,sin(alpha_x),cos(alpha_x)];
%   
% Ry = [cos(alpha_y),0,sin(alpha_y);
%       0,1,0;
%       -sin(alpha_y),0,cos(alpha_y)];
%   
% Rz = [cos(alpha_z),-sin(alpha_z),0;
%       sin(alpha_z),cos(alpha_z),0;
%       0,0,1];
%   
%   R = Rx*Ry*Rz;
    
    Points_A = [Points_A;basic_ring*RM+[x1(i),y1(i),z1(i)]];

end


R = 0.2;
Y = 0.1;
t = pi:0.04*pi:5*pi
x1 = R*cos(t);
y1 = R*sin(t);
z1 = Y*(t-pi);


theta = 0:0.1*pi:2*pi;
ring_x = r*cos(theta');
ring_y = r*sin(theta');

basic_ring = [ring_x,ring_y,zeros(length(theta),1)];

Points_B = [];

for i = 1:length(t)
    
    alpha_x = -R*sin(t(i))/sqrt(R^2+Y^2);
    alpha_y = R*cos(t(i))/sqrt(R^2+Y^2);
    alpha_z = Y/sqrt(R^2+Y^2);
    
    A = [alpha_x, alpha_y, alpha_z];
    B = [0,0,1];
    v = cross(A,B);
    s = sqrt(v(1)^2+v(2)^2+v(3)^2);
    v_x = [0, -v(3),v(2);
           v(3),0,-v(1);
           -v(2),v(1),0];
    
    c = A*B';
    RM = eye(3)+v_x + v_x*v_x*(1-c)/s^2;

%      Rx = [1,0,0;
%       0,cos(alpha_x),-sin(alpha_x);
%       0,sin(alpha_x),cos(alpha_x)];
%   
% Ry = [cos(alpha_y),0,sin(alpha_y);
%       0,1,0;
%       -sin(alpha_y),0,cos(alpha_y)];
%   
% Rz = [cos(alpha_z),-sin(alpha_z),0;
%       sin(alpha_z),cos(alpha_z),0;
%       0,0,1];
%   
%   R = Rx*Ry*Rz;
    
    Points_B = [Points_B;basic_ring*RM+[x1(i),y1(i),z1(i)]];

end

Points_DoubleSpiral_A = Points_A;
Points_DoubleSpiral_B = Points_B;

c_A = 0.2*(Points_A(:,3)-min(Points_A(:,3)))./(max(Points_A(:,3))-min(Points_A(:,3)));
c_B = 0.2*(Points_B(:,3)-min(Points_B(:,3)))./(max(Points_B(:,3))-min(Points_B(:,3)))+0.4


figure()

scatter3(Points_A(:,1),Points_A(:,2),Points_A(:,3),30,c_A, 'filled')

hold on

scatter3(Points_B(:,1),Points_B(:,2),Points_B(:,3),30,c_B, 'filled')

colormap('jet')

axis equal

axis off

%%
Train_DNA_1_A = Points_A;
Train_DNA_1_B = Points_B;


dlmwrite('Train_DNA_1_A.txt', Train_DNA_1_A, 'delimiter',' ')
dlmwrite('Train_DNA_1_B.txt', Train_DNA_1_B, 'delimiter',' ')

