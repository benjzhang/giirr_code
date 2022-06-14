function output = gradlogpos_gg_riem(thet,GxX,Gp,c,n)
N = size(GxX,2); 
index = randsample(N,n); 
mp = N/n * (Gp\sum(GxX(:,index),2)) ;
% Gp = Gthet + N * Gx ;

output = -c*( thet - mp) ;


end