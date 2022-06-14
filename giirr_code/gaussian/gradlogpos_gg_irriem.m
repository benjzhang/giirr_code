function output = gradlogpos_gg_irriem(thet,GxX,Gp,J,c,n)
N = size(GxX,2); 
index = randsample(N,n); 
mp = N/n * (Gp\sum(GxX(:,index),2) );
% Gp = Gthet + N * Gx ;

matrix = eye(3) + J + Gp\(J*Gp) ; 
output = -c * matrix * (thet - mp) ;


end