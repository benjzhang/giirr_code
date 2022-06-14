%%% Independent Component Analysis with irreversible SGLD


rng('shuffle') ;

% load('ica_data.mat') ;
load('ica_data3.mat');
% Bayesian inference setup


% Prior precision
lambda = 1 ;
T = 1000;
dt = 0.2e-2/100;

% delta =  3 ; % delta = 3 max
% C0 = kron(eye(d),C00) + kron(C00,eye(d)) ;
% const = delta/norm(C0) ;
% J = C0 * const ;
% J = triu(ones(d^2)) ;
% J = J-J' ;
% J = delta * J /norm(J) ;
%
J = J*0.5; C00 =  C00*0.5; 
divergence_symbolic

for ii = 1000:1000 % 8 and on are new J
    W_irriem = zeros(d^2,round(T/dt)) ;
    W_irriem(:,1) = reshape(diag((rand(3,1)>0.5)*2-1),d^2,1) ;
    Wold_ir = W_irriem(:,1) ;
    
    for jj = 1:T/dt
        
        Wold_sqir = reshape(Wold_ir,d,d);
        Wnew_ir = Wold_ir + dt * gradlogpos_irr_riem_new(Wold_ir,X,C00,lambda)/2 + sqrt(dt) * reshape(randn(d) * real(sqrtm(eye(d)+ Wold_sqir' * Wold_sqir)),d^2,1)  ;
        W_irriem(:,jj+1) = Wnew_ir ;
        Wold_ir = Wnew_ir ;
        
    end
    
    Wss_irriem = W_irriem(:,1:500:end) ;
    obs1 = sum(Wss_irriem,1);
    obs2 = sum(Wss_irriem.^2,1) ;
    
    name = sprintf('irriemobs%i_new.mat',ii);
    save(name,'obs1','obs2','Wss_irriem','dt');
% obs1 = obs1(500001:end) ;
% obs2 = obs2(500001:end) ;
%       save(name,'obs1','obs2','dt');    
end

% %
% Wn_subsampled_ir = W_ir(:,1:10:end) ;
% save('wnsublong_ir.mat','Wn_subsampled_ir') ;
% mupost = mean(W(:,20/dt+1:end),2) ;
% Wpost = reshape(mupost,d,d) ;
%
exit