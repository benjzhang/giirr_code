%%% Independent Component Analysis with irreversible SGLD


rng('shuffle') ;

% Bayesian inference setup
load('ica_data3.mat');


% Prior precision
lambda = 1 ;
T = 2000;
dt = 0.2e-2/100;

for ii = 1:100
    W_riem = zeros(d^2,round(T/dt)) ;
    
    W_riem(:,1) = reshape(diag((rand(3,1)>0.5)*2-1),d^2,1) ;
    Wold_riem = W_riem(:,1) ;
    
    
    
    for jj = 1:T/dt-1
        %
        Wold_sq = reshape(Wold_riem,d,d) ;
        Wnew_riem = Wold_riem + dt * gradlogpos_riem_new(Wold_riem,X,lambda)/2 + sqrt(dt) * reshape(randn(d) * real(sqrtm(eye(d) + Wold_sq' * Wold_sq)),d^2,1)  ;
        W_riem(:,jj+1) = Wnew_riem ;
        Wold_riem = Wnew_riem ;
    end
    
    
    Wss_riem = W_riem(:,1:500:end) ;
    obs1 = sum(Wss_riem,1);
    obs2 = sum(Wss_riem.^2,1) ;
    
    name = sprintf('riemobs%i_2.mat',ii);
    save(name,'obs1','obs2','Wss_riem','dt');
% obs1 = obs1(500001:end) ;
% obs2 = obs2(500001:end) ;
%       save(name,'obs1','obs2','dt');
end
% %
% Wn_subsampled_riem = W_riem(:,1:10:end) ;
% save('wnsublong_riem.mat','Wn_subsampled_riem') ;
% mupost = mean(W(:,20/dt+1:end),2) ;
% Wpost = reshape(mupost,d,d) ;
%
