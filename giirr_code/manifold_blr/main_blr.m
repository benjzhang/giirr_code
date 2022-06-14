% Bayesian logistic regression

%% load the data
load('benchmarks','german') ;
set = german; 
train = set.test ;
test = set.train ;
ind = 42 ;
xtrain = set.x(train(:,ind),:) ;
ttrain = set.t(train(:,ind),:) ;
xtest = set.x(test(:,ind),:) ;
ttest = set.t(test(:,ind),:) ;
ttrain = (ttrain== 1) ;
ttest = (ttest == 1) ;


%% parameters
rng(128423)
h = 2.5e-4;
alpha = 1 ;
Nsteps = 400000 ;
Ntraj = 100 ;
T = h * Nsteps ;
t = 0:h:T ;

d = size(xtrain,2); 
X = xtrain' ;
% delta = 1; 
% J = delta * [0 1; -1 0] ;
% J = delta * triu(sign(randn(d))) ;
% J = J- J' ;
% Jinv = inv(J) ;
% load('Jinfo20'); 

%% Langevin

n = 10 ;
for kk = 1:10
w1 = zeros(d,Nsteps,Ntraj) ;
parfor ii = 1:Ntraj
    wtemp = zeros(d,Nsteps) ;
    for jj = 1:Nsteps-1
        wtemp(:,jj+1)  = wtemp(:,jj) + 0.5 * h * gradlogpos_blr(wtemp(:,jj),alpha,X,ttrain,n) + sqrt(h) * randn(d,1) ;
    end
    w1(:,:,ii) = wtemp ;
end
w1first = sum(w1,1); 
w1sec = sum(w1.^2,1); 
name = sprintf('lang20_1%i.mat',kk); 
save(name,'w1first','w1sec');
end
exit;

% % exit; 



Logpos = @(w) -alpha * norm(w)^2 /2 + dot(ttrain,xtrain * w) - sum(log(1+exp(xtrain*w))) ;

wnow = zeros(d,1); propstd = 0.3 ;
Nlength = 1e6;
wchain = zeros(d,Nlength) ;
Nacc = 0;
for ii = 1:Nlength
    
    wprop = wnow + propstd * randn(d,1) ;
    ratio = exp(Logpos(wprop) - Logpos(wnow) );
    if ratio > rand
        wnow = wprop;
        Nacc = Nacc + 1 ;
    end
    
    
    wchain(:,ii) = wnow ;
    
end

mupost = mean(wchain(:,10000:end),2);
secmom = mean(wchain(:,10000:end).^2,2);
[mse0first,mse0sec,bias0first,bias0sec] = postprocess(w0,secmom,mupost,5/h) ;
[mse1first,mse1sec,bias1first,bias1sec] = postprocess(w1,secmom,mupost,5/h) ;
% [mse2first,mse2sec,bias2first,bias2sec] = postprocess(w2,secmom,mupost,5/h) ;
% [mse3first,mse3sec,bias3first,bias3sec] = postprocess(w3,secmom,mupost,5/h) ;
% [mse4first,mse4sec,bias4first,bias4sec] = postprocess(w4,secmom,mupost,5/h) ;
% [mse5first,mse5sec,bias5first,bias5sec] = postprocess(w5,secmom,mupost,5/h) ;

% 
% %% Riemmanian Langevin
% w2 = zeros(d,Nsteps,Ntraj) ;
% 
% parfor ii = 1:Ntraj
%     wtemp = zeros(d,Nsteps) ;
%     for jj = 1:Nsteps-1
%         G =  metric(X,wtemp(:,jj),alpha); 
%         Gsq = sqrtm(G) ;
%         wtemp(:,jj+1)  = wtemp(:,jj) + 0.5 * h * gradlogpos_blr_riem(wtemp(:,jj),alpha,X,ttrain,n,G) + sqrt(h) * (Gsq\ randn(d,1)) ;
%     end
%     w2(:,:,ii) = wtemp ;
% end
% 
% 
% 
% %% Irr
% w3 = zeros(d,Nsteps,Ntraj) ;
% parfor ii = 1:Ntraj
%     wtemp = zeros(d,Nsteps) ;
%     for jj = 1:Nsteps-1
%         wtemp(:,jj+1)  = wtemp(:,jj) + 0.5 * (eye(d) + J) * h * gradlogpos_blr(wtemp(:,jj),alpha,X,ttrain,n) + sqrt(h) * randn(d,1) ;
%     end
%     w3(:,:,ii) = wtemp ;
% end
% 
% 
% %% Riemmanian Langevin with irrev
% w4 = zeros(d,Nsteps,Ntraj) ;
% 
% parfor ii = 1:Ntraj
%     wtemp = zeros(d,Nsteps) ;
%     for jj = 1:Nsteps-1
%         G =  metric(X,wtemp(:,jj),alpha); 
%         Gsq = sqrtm(G) ;
%         wtemp(:,jj+1)  = wtemp(:,jj) + 0.5 * h * gradlogpos_blr_riem(wtemp(:,jj),alpha,X,ttrain,n,G)+...
%                             0.5 * h * J * gradlogpos_blr(wtemp(:,jj),alpha,X,ttrain,n)+ sqrt(h) * (Gsq\ randn(d,1)) ;
%     end
%     w4(:,:,ii) = wtemp ;
% end
% 
% 
% 
% % %% Geometry-informed
% w5 = zeros(d,Nsteps,Ntraj) ;
% 
% parfor ii = 1:Ntraj
%     wtemp = zeros(d,Nsteps) ;
%     for jj = 1:Nsteps-1
%         G =  metric(X,wtemp(:,jj),alpha); 
%         Gsq = sqrtm(G) ;
%         wtemp(:,jj+1)  = wtemp(:,jj) + 0.5 * h * gradlogpos_blr_irriem(wtemp(:,jj),alpha,X,ttrain,n,G,0.5*J,2*Jinv) + sqrt(h) * (Gsq\ randn(d,1)) ;
%     end
%     w5(:,:,ii) = wtemp ;
% end
% 

% % SGLD
% n = 30 ;
% w2 = zeros(2,Nsteps,Ntraj) ;
% parfor ii = 1:Ntraj
%     wtemp = -ones(2,Nsteps) ;
%     for jj = 1:Nsteps-1
%         wtemp(:,jj+1)  = langevin_sgld_step_blr(wtemp(:,jj),alpha,xtrain,ttrain,n,h) ;
%     end
%     w2(:,:,ii) = wtemp ;
% end
% 
% 
% % Irr_SGLD
% delta = 2 ;
% w3 = zeros(2,Nsteps,Ntraj) ;
% C0 = [0 1; -1 0];
% C = delta * C0  ;
% 
% parfor ii = 1:Ntraj
%     wtemp = -ones(2,Nsteps) ;
%     for jj = 1:Nsteps-1
%         wtemp(:,jj+1)  = langevin_sgld_step_blr(wtemp(:,jj),alpha,xtrain,ttrain,n,h,C) ;
%     end
%     w3(:,:,ii) = wtemp ;
% end



% %% Postprocessing
% est1 = zeros(2,Nsteps,Ntraj) ;
% mse1 = zeros(1,Nsteps,Ntraj) ;
% mupost = [ -0.106 ; -0.079] ;
% for ii = 1:Ntraj
%     est1(:,:,ii) = cumsum(w1(:,:,ii),2) ./ repmat(1:Nsteps,2,1);
%     mse1(1,:,ii) = dot(est1(:,:,ii) - mupost,est1(:,:,ii) - mupost)  ;
% end
% mse1 = mean(mse1,3) ;
% 
% est2 = zeros(2,Nsteps,Ntraj) ;
% mse2 = zeros(1,Nsteps,Ntraj) ;
% for ii = 1:Ntraj
%     est2(:,:,ii) = cumsum(w2(:,:,ii),2) ./ repmat(1:Nsteps,2,1);
%     mse2(1,:,ii) = dot(est2(:,:,ii) - mupost,est2(:,:,ii) - mupost)  ;
% end
% mse2 = mean(mse2,3) ;
% 
% est3 = zeros(2,Nsteps,Ntraj) ;
% mse3 = zeros(1,Nsteps,Ntraj) ;
% for ii = 1:Ntraj
%     est3(:,:,ii) = cumsum(w3(:,:,ii),2) ./ repmat(1:Nsteps,2,1);
%     mse3(1,:,ii) = dot(est3(:,:,ii) - mupost,est3(:,:,ii) - mupost)  ;
% end
% mse3 = mean(mse3,3) ;


%% Plotting

%% Plot posterior
% x = -2:0.01:2 ;
% y = x ;
% z = zeros(length(x),length(y)) ;
% 
% for ii = 1:length(x)
%     for jj = 1:length(y)
%         z(ii,jj) = -alpha/2 * (x(ii)^2+y(jj)^2) + dot(ttrain,xtrain * [x(ii);y(jj)]) - ...
%             sum(log(1+exp(xtrain*[x(ii);y(jj)])));
%     end
% end
% 

Logpos = @(w) -alpha * norm(w)^2 /2 + dot(ttrain,xtrain * w) - sum(log(1+exp(xtrain*w))) ;

wnow = zeros(d,1); propstd = 0.3 ;
Nlength = 250000;
wchain = zeros(d,Nlength) ;
Nacc = 0; 
for ii = 1:Nlength
    
    wprop = wnow + propstd * randn(d,1) ;
    ratio = exp(Logpos(wprop) - Logpos(wnow) );
    if ratio > rand
        wnow = wprop;
        Nacc = Nacc + 1 ;
    end
    
    
    wchain(:,ii) = wnow ;
    
end

% 
% 
% 
% 
% 




