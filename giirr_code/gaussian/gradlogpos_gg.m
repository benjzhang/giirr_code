function output = gradlogpos_gg(thet,GxX,Gp,n)
N = size(GxX,2); 
index = randsample(N,n); 
mp = N/n * sum(GxX(:,index),2) ;
% Gp = Gthet + N * Gx ;

output = - Gp * thet + mp;


end