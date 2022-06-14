%%% Independent Component Analysis with irreversible SGLD


rng('shuffle') ;

% load('ica_data.mat') ;
load('ica_data3.mat');
% Bayesian inference setup
% Prior precision
lambda = 1 ;
T = 1000;
dt = 0.2e-4;
% 
% delta =  10 ; % delta = 3 max
% C0 = kron(eye(d),C00) + kron(C00,eye(d)) ;
% const = delta/norm(C0) ;
% J = C0 * const ;
% 
J = 2*J * 0.5 ;


for ii = 1000:1000 %nine and on
    
W_irwriem = zeros(d^2,round(T/dt)) ;
W_irwriem(:,1) = reshape(diag((rand(3,1)>0.5)*2-1),d^2,1);
Wold = W_irwriem(:,1) ;

    for jj = 1:T/dt-1
        % % %
        Wold_sq = reshape(Wold,d,d) ;
        Wnew = Wold + dt *  gradlogpos_irrwriem_new(Wold,X,J,lambda) /2 + sqrt(dt) * reshape(randn(d) * real(sqrtm(eye(d) + Wold_sq' * Wold_sq)),d^2,1) ;
        W_irwriem(:,jj+1) = Wnew ;
        Wold = Wnew ;
        
    end
    
    Wss_irwriem = W_irwriem(:,1:500:end) ;
    obs1 = sum(Wss_irwriem,1);
    obs2 = sum(Wss_irwriem.^2,1) ;
    
    name = sprintf('irwriemobs%i_new.mat',ii);
    save(name,'obs1','obs2','Wss_irwriem','J','dt');
% obs1 = obs1(500001:end) ;
% obs2 = obs2(500001:end) ;
%       save(name,'obs1','obs2','dt');
end

%
% Wss_irwriem = W_irwriem(:,1:10:end) ;
% save('Wss_irwriem2.mat','Wss_irwriem','dt','delta');

%
% Wn_subsampled = Wn(:,1:10:end) ;
% save('wnsublong.mat','Wn_subsampled','dt') ;
% % mupost = mean(W(:,20/dt+1:end),2) ;
% % Wpost = reshape(mupost,d,d) ;
% %
exit
