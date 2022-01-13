

r_ring = 0.2;
R_ring = 1;
t = 0:0.02*pi:8*pi;
x1 = cos(t).*(r_ring*sin(8*t)+R_ring);
y1 = sin(t).*(r_ring*sin(8*t)+R_ring);
z1 = r_ring*cos(8*t);



r = 0.05;
theta = 0:0.4*pi:2*pi;
ring_x = r*cos(theta');
ring_y = r*sin(theta');

basic_ring = [ring_x,ring_y,zeros(length(theta),1)];

Points_B = [];

for i = 1:length(t)
    
    alpha_x = -R_ring*sin(t(i))+4*r_ring*cos(4*t(i));
    alpha_y =  R_ring*cos(t(i))-4*r_ring*sin(4*t(i));
    alpha_z = -4*r_ring*cos(4*t(i));
    
    alpha_x = alpha_x/sqrt(alpha_x^2+ alpha_y^2+alpha_z^2);
    alpha_y = alpha_y/sqrt(alpha_x^2+ alpha_y^2+alpha_z^2);
    alpha_z = alpha_z/sqrt(alpha_x^2+ alpha_y^2+alpha_z^2);
      
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

alpha_z = 0.1*pi;

Rz = [cos(alpha_z),-sin(alpha_z),0;
      sin(alpha_z),cos(alpha_z),0;
      0,0,1];

Points_A = Points_B*Rz;

Points_TwoCurlySpiral_A = Points_A;
Points_TwoCurlySpiral_B = Points_B;

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
Train_TwoCurlySpirals_A = Points_A;
Train_TwoCurlySpirals_B = Points_B;


dlmwrite('Train_TwoCurlySpirals_A.txt', Train_TwoCurlySpirals_A, 'delimiter',' ')
dlmwrite('Train_TwoCurlySpirals_B.txt', Train_TwoCurlySpirals_B, 'delimiter',' ')
