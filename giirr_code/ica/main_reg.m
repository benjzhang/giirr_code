%%% Independent Component Analysis with irreversible SGLD


rng('shuffle') ;
% load('ica_data.mat') ;
load('ica_data3.mat');
% Bayesian inference setup
% Prior precision
lambda = 1 ;
T = 2000;
dt = 0.2e-4;

for ii = 1:39
    Wn = zeros(d^2,round(T/dt)) ;
    Wn(:,1) = reshape(diag((rand(3,1)>0.5)*2-1),d^2,1);
    Wold = Wn(:,1) ;
    
    for jj = 1:T/dt-1
        % % %
        Wnew = Wold + dt * gradlogpos(Wold,X,lambda) /2 + sqrt(dt) * randn(d^2,1) ;
        Wn(:,jj+1) = Wnew ;
        Wold = Wnew ;
        
    end
    
    
    Wss_n = Wn(:,1:500:end) ;
    obs1 = sum(Wss_n,1);
    obs2 = sum(Wss_n.^2,1) ;
    
    name = sprintf('n%i.mat',ii);
%     save(name,'obs1','obs2','Wss_n','dt');
% obs1 = obs1(500001:end) ;
% obs2 = obs2(500001:end) ;
      save(name,'obs1','obs2','dt');

end
%
% Wss_n = Wn(:,1:10:end) ;
% save('Wss_n.mat','Wss_n','dt');
%
% Wn_subsampled = Wn(:,1:10:end) ;
% save('wnsublong.mat','Wn_subsampled','dt') ;
% % mupost = mean(W(:,20/dt+1:end),2) ;
% % Wpost = reshape(mupost,d,d) ;
% %
exit
