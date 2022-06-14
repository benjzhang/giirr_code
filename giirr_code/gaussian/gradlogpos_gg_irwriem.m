function output = gradlogpos_gg_irwriem(thet,GxX,Gp,J,c,n)
N = size(GxX,2); 
index = randsample(N,n); 
mp = N/n * sum(GxX(:,index),2) ;
% Gp = Gthet + N * Gx ;

output = - c * (thet - Gp\mp) - J * (Gp * thet - mp) ;


end