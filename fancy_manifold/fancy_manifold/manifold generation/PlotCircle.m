R = 0.4;

t = 0:0.04*pi:2*pi
x1 = R*cos(t);
y1 = R*sin(t);
z1 = 0*t;

r = 0.04;
theta = 0:0.2*pi:2*pi;
ring_x = r*cos(theta');
ring_y = r*sin(theta');

basic_ring = [ring_x,ring_y,zeros(length(theta),1)];

   
Points1 = [];

for i = 1:length(t)
    
    alpha_x = -sin(t(i));
    alpha_y = cos(t(i));
    alpha_z = 0;
    
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
    
    Points1 = [Points1;basic_ring*RM+[x1(i),y1(i),z1(i)]];

end





Points2 = Points1;
Points2(:,1) = Points2(:,1)+0.8;

Points3 = Points1;
Points3(:,1) = Points3(:,1)+0.45;
Points3(:,2) = Points3(:,2)+0.7;

%%
alpha_x = 0.5*pi;

Rx = [1,0,0;
      0,cos(alpha_x),-sin(alpha_x);
      0,sin(alpha_x),cos(alpha_x)];
  
  

Points4 = Points1*Rx;
Points4(:,1) = Points4(:,1)+0.45;

alpha_y = 0.5*pi;

Ry = [cos(alpha_y),0,sin(alpha_y);
      0,1,0;
      -sin(alpha_y),0,cos(alpha_y)];

alpha_z = -0.3*pi;

Rz = [cos(alpha_z),-sin(alpha_z),0;
      sin(alpha_z),cos(alpha_z),0;
      0,0,1];

Points5 = Points4*Rz;
Points5(:,1) = Points5(:,1)-0.1;


alpha_z = 0.3*pi;

Rz = [cos(alpha_z),-sin(alpha_z),0;
      sin(alpha_z),cos(alpha_z),0;
      0,0,1];

Points6 = Points4*Rz;

Points6(:,1) = Points6(:,1)+0.4;
Points6(:,2) = Points6(:,2)+0.72;

Points4(:,1) = Points4(:,1) -0.05;

%%
Points_A = [Points1;Points2;Points3]+0.7;
Points_B = [Points4;Points5;Points6]+0.7;


Points_Circle_A = Points_A;
Points_Circle_B = Points_B;


c_A = 0.2*(Points_A(:,3)-min(Points_A(:,3)))./(max(Points_A(:,3))-min(Points_A(:,3)));
c_B = 0.2*(Points_B(:,3)-min(Points_B(:,3)))./(max(Points_B(:,3))-min(Points_B(:,3)))+0.5;


figure()

scatter3(Points_A(:,1),Points_A(:,2),Points_A(:,3),30,c_A, 'filled')
hold on

scatter3(Points_B(:,1),Points_B(:,2),Points_B(:,3),30,c_B, 'filled')

colormap('jet')
axis equal
axis off

%%

Train_Circle_A = Points_A;
Train_Circle_B = Points_B;


dlmwrite('Train_Circle_A.txt', Train_Circle_A, 'delimiter',' ')
dlmwrite('Train_Circle_B.txt', Train_Circle_B, 'delimiter',' ')


