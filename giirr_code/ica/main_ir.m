%%% Independent Component Analysis with irreversible SGLD


rng('shuffle') ;

% Bayesian inference setup
load('ica_data3.mat');

% Prior precision
lambda = 1 ;
T = 2000;
dt = 0.2e-4;

% delta =  6 ; % delta = 3 max
% C0 = kron(eye(d),C00) + kron(C00,eye(d)) ;
% const = delta/norm(C0) ;
% J = C0 * const ;

J = 2*J * 0.5 ;
for ii = 1:100 %different beyond 6
    W_ir = zeros(d^2,round(T/dt)) ;
    W_ir(:,1) = reshape(diag((rand(3,1)>0.5)*2-1),d^2,1);
    Wold = W_ir(:,1) ;
    
    for jj = 1:T/dt-1
        % % %
        Wnew = Wold + dt * (eye(d^2) + J) *  gradlogpos(Wold,X,lambda) /2 + sqrt(dt) * randn(d^2,1) ;
        W_ir(:,jj+1) = Wnew ;
        Wold = Wnew ;
        
    end
    
    Wss_ir = W_ir(:,1:500:end) ;
    obs1 = sum(Wss_ir,1);
    obs2 = sum(Wss_ir.^2,1) ;
    
    name = sprintf('ir%i_2.mat',ii);
    save(name,'obs1','obs2','Wss_ir','dt','J');
% obs1 = obs1(500001:end) ;
% obs2 = obs2(500001:end) ;
%       save(name,'obs1','obs2','dt');
end
% save('Wss_ir2.mat','Wss_ir','dt','delta');
%
% Wn_subsampled = Wn(:,1:10:end) ;
% save('wnsublong.mat','Wn_subsampled','dt') ;
% % mupost = mean(W(:,20/dt+1:end),2) ;
% % Wpost = reshape(mupost,d,d) ;
% %
exit
