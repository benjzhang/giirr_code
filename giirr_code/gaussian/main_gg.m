% 2D Gaussian-Gaussian case
rng(1241243); 
% parameters
d = 3; 
theta_true = [2; -1; 1] ; % true mean
Gammathet = diag([1,1,1]) * 0.25 ; % prior std dev
Gammax = randn(3); 
[a,b] = qr(Gammax); 
Gammax = a * diag([0.2;0.01;0.05]) * a' ;

N = 10 ; % Number of samples
h = 0.005; % stepsize
T = 505; 

M = 1000 ; 

Nsteps = T/h + 1 ;
Ntraj = M ;

% Data
X = mvnrnd(theta_true,inv(Gammax),N)' ;

% true post
Gammap =  Gammathet + N * Gammax ;

c = 1; 
Gpinv = c *eye(3) /Gammap; 
sqrtGpinv = sqrtm(c * Gpinv); 


mupost = Gammap\Gammax * sum(X,2) ;
% sigmapost2 = 1/(1/st^2 + N/sx^2) ; 
GammaxX = Gammax * X ;


% Irrev
delta = 1; 
J = delta * [0 1 1; -1 0 1 ; -1 -1 0]; 
n = 2; 
rng('shuffle') ;


% Langevin
t0 = zeros(d,Nsteps,M) ;

parfor jj = 1:M
thet0 = zeros(d,Nsteps) ;
for ii = 1:Nsteps-1
    
    thet0(:,ii+1) = thet0(:,ii) + h/2 * gradlogpos_gg(thet0(:,ii),GammaxX,Gammap,N) + sqrt(h) * randn(d,1); 
end
t0(:,:,jj) = thet0 ;
end


% SGLD
t1 = zeros(d,Nsteps,M); 
parfor jj = 1:M
thet1 = zeros(d,Nsteps) ;
for ii = 1:Nsteps-1
    
    thet1(:,ii+1) = thet1(:,ii) + h/2 * gradlogpos_gg(thet1(:,ii),GammaxX,Gammap,n) + sqrt(h) * randn(d,1); 
end
t1(:,:,jj) = thet1 ;
end

% Riemmanian Langevin
t2 = zeros(d,Nsteps,M); 
parfor jj = 1:M
thet2 = zeros(d,Nsteps); 
for ii = 1:Nsteps-1
    
    thet2(:,ii+1) = thet2(:,ii) + h/2 * gradlogpos_gg_riem(thet2(:,ii),GammaxX,Gammap,c,n) + sqrt(h) * sqrtGpinv * randn(d,1); 
end
t2(:,:,jj) = thet2; 

end

%% Irreversible
t3 = zeros(d,Nsteps,M); 
parfor jj = 1:M
thet3 = zeros(d,Nsteps); 
for ii = 1:Nsteps-1
    
    thet3(:,ii+1) = thet3(:,ii) + h/2 * (eye(d) +J) * gradlogpos_gg(thet3(:,ii),GammaxX,Gammap,n) + sqrt(h) * randn(d,1); 
end
t3(:,:,jj) = thet3 ;
end


% Irreversible to riem
t4 = zeros(d,Nsteps,M); 
parfor jj = 1:M
thet4 = zeros(d,Nsteps); 
for ii = 1:Nsteps-1
    
    thet4(:,ii+1) = thet4(:,ii) + h/2 * gradlogpos_gg_irwriem(thet4(:,ii),GammaxX,Gammap,J,c,n) + sqrt(h) * sqrtGpinv * randn(d,1); 
end
t4(:,:,jj) = thet4; 
end


% geometry adapted
t5 = zeros(d,Nsteps,M); 
parfor jj = 1:M
thet5 = zeros(d,Nsteps); 
for ii = 1:Nsteps-1
    
    thet5(:,ii+1) = thet5(:,ii) + h/2 * gradlogpos_gg_irriem(thet5(:,ii),GammaxX,Gammap,J/2,c,n) + sqrt(h) * sqrtGpinv * randn(d,1); 
end
t5(:,:,jj) = thet5; 
end




%% Postprocessing
secmom = diag(inv(Gammap))+mupost.^2 ;
% 
burnIn = 5/h+1 ;
% est0first = zeros(3,Nsteps-burnIn+1,Ntraj) ;
% est0sec = est0first; 
% 
% mse0first = zeros(1,Nsteps-burnIn+1,Ntraj) ;
% mse0sec = mse0first ;
% 
% for ii = 1:Ntraj
%     est0first(:,:,ii) = cumsum(t0(:,burnIn:end,ii),2)./repmat(1:Nsteps-burnIn+1,3,1) ;
% est0sec(:,:,ii) = cumsum(t0(:,burnIn:end,ii).^2,2)./ repmat(1:Nsteps-burnIn+1,3,1);
% est0firsttemp = sum(est0first(:,:,ii),1); 
% est0sectemp = sum(est0sec(:,:,ii),1); 
% % mse0(1,:,ii) = dot(est0(:,:,ii) - secmom,est0(:,:,ii) - secmom)  ;
% mse0first(1,:,ii) = (est0firsttemp - sum(mupost)) .^2; 
% mse0sec(1,:,ii) = (est0sectemp - sum(secmom)).^2 ;
% 
% end
% mse0first = mean(mse0first,3) ;
% mse0sec = mean(mse0sec,3); 
% 
% bias0first = (mean(sum(est0first,1),3)-sum(mupost)).^2 ;
% bias0sec = (mean(sum(est0sec,1),3)-sum(secmom)).^2 ;
% 
% 

[mse0first,mse0sec,bias0first,bias0sec] = postprocess(t0,secmom,mupost,burnIn); 
[mse1first,mse1sec,bias1first,bias1sec] = postprocess(t1,secmom,mupost,burnIn); 
[mse3first,mse3sec,bias3first,bias3sec] = postprocess(t3,secmom,mupost,burnIn); 
[mse2first,mse2sec,bias2first,bias2sec] = postprocess(t2,secmom,mupost,burnIn); 
[mse4first,mse4sec,bias4first,bias4sec] = postprocess(t4,secmom,mupost,burnIn); 
[mse5first,mse5sec,bias5first,bias5sec] = postprocess(t5,secmom,mupost,burnIn); 
% 
% 
% est1 = zeros(3,Nsteps-burnIn+1,Ntraj) ;
% mse1 = zeros(1,Nsteps-burnIn+1,Ntraj) ;
% for ii = 1:Ntraj
% est1(:,:,ii) = cumsum(t1(:,burnIn:end,ii),2) ./ repmat(1:Nsteps-burnIn+1,3,1);
% est1(:,:,ii) = sum(est1,1);
% mse1(1,:,ii) = (est1 - sum(mupost)).^2 ;
% 
% % mse1(1,:,ii) = dot(est1(:,:,ii) - mupost,est1(:,:,ii) - mupost)  ;
% 
% end
% mse1 = mean(mse1,3) ;
% 
% 
% 
% est2 = zeros(3,Nsteps-burnIn+1,Ntraj) ;
% mse2 = zeros(1,Nsteps-burnIn+1,Ntraj) ;
% for ii = 1:Ntraj
% est2(:,:,ii) = cumsum(t2(:,burnIn:end,ii),2) ./ repmat(1:Nsteps-burnIn+1,3,1);
% mse2(1,:,ii) = dot(est2(:,:,ii) - mupost,est2(:,:,ii) - mupost)  ;
% end
% mse2 = mean(mse2,3) ;
% 
% est3 = zeros(3,Nsteps-burnIn+1,Ntraj) ;
% mse3 = zeros(1,Nsteps-burnIn+1,Ntraj) ;
% for ii = 1:Ntraj
% est3(:,:,ii) = cumsum(t3(:,burnIn:end,ii),2) ./ repmat(1:Nsteps-burnIn+1,3,1);
% mse3(1,:,ii) = dot(est3(:,:,ii) - mupost,est3(:,:,ii) - mupost)  ;
% end
% mse3 = mean(mse3,3) ;
% 
% est4 = zeros(3,Nsteps-burnIn+1,Ntraj) ;
% mse4 = zeros(1,Nsteps-burnIn+1,Ntraj) ;
% for ii = 1:Ntraj
% est4(:,:,ii) = cumsum(t4(:,burnIn:end,ii),2) ./ repmat(1:Nsteps-burnIn+1,3,1);
% mse4(1,:,ii) = dot(est4(:,:,ii) - mupost,est4(:,:,ii) - mupost)  ;
% end
% mse4 = mean(mse4,3) ;
% 
% 
% est5 = zeros(3,Nsteps-burnIn+1,Ntraj) ;
% mse5 = zeros(1,Nsteps-burnIn+1,Ntraj) ;
% for ii = 1:Ntraj
% est5(:,:,ii) = cumsum(t5(:,burnIn:end,ii),2) ./ repmat(1:Nsteps-burnIn+1,3,1);
% mse5(1,:,ii) = dot(est5(:,:,ii) - mupost,est5(:,:,ii) - mupost)  ;
% end
% mse5 = mean(mse5,3) ;
% % 
% % est2 = zeros(2,Nsteps,Ntraj) ;
% % mse2 = zeros(1,Nsteps,Ntraj) ;
% for ii = 1:Ntraj
% est2(:,:,ii) = cumsum(thet2(:,:,ii),2) ./ repmat(1:Nsteps,2,1);
% mse2(1,:,ii) = dot(est2(:,:,ii) - mupost,est2(:,:,ii) - mupost)  ;
% end
% mse2 = mean(mse2,3) ;
% 
% est3 = zeros(2,Nsteps,Ntraj) ;
% mse3 = zeros(1,Nsteps,Ntraj) ;
% for ii = 1:Ntraj
% est3(:,:,ii) = cumsum(thet3(:,:,ii),2) ./ repmat(1:Nsteps,2,1);
% mse3(1,:,ii) = dot(est3(:,:,ii) - mupost,est3(:,:,ii) - mupost)  ;
% end
% mse3 = mean(mse3,3) ;
% 
% 
% 
% figure(1); 
% plot(mse1,'linewidth',2); 
% hold on
% plot(mse2,'linewidth',2) ;
% plot(mse3,'linewidth',2) ;
% legend({'Langevin', 'SGLD', 'Irr. SGLD'},'interpreter','latex') ;
% grid on
% set(gca,'fontsize',18) ;
% set(gca,'TickLabelInterpreter','latex') ;
% xlabel('Steps','interpreter','latex') ;
% ylabel('MSE','interpreter','latex') ;
% 
% 


figure(1) ;
subplot(1,3,1) ;
loglog(mse1first,'linewidth',2) ;
hold on
loglog(mse2first,'linewidth',2) ;
loglog(mse3first,'linewidth',2) ;
loglog(mse4first,'linewidth',2) ;
loglog(mse5first,'linewidth',2) ; grid on
set(gca,'fontsize',24) ;
set(gca,'TickLabelInterpreter','latex') ;
xlabel('$K$','interpreter','latex') ;
ylabel('MSE','interpreter','latex') ;
hold off ; 
legend({'LD','RM','Irr','RMIrr','GiIrr'},'interpreter','latex') ;


subplot(1,3,2) ;
loglog(bias1first,'linewidth',2) ;
hold on
loglog(bias2first,'linewidth',2) ;
loglog(bias3first,'linewidth',2) ;
loglog(bias4first,'linewidth',2) ;
loglog(bias5first,'linewidth',2) ; grid on
set(gca,'fontsize',24) ;
set(gca,'TickLabelInterpreter','latex') ;
xlabel('$K$','interpreter','latex') ;
ylabel('Bias$^2$','interpreter','latex') ;
title('$\phi(\theta) = \theta_1+\theta_2+\theta_3$','interpreter','latex'); 
hold off  


var1first = mse1first - bias1first ;
var2first = mse2first - bias2first ;
var3first = mse3first - bias3first ;
var4first = mse4first - bias4first ;
var5first = mse5first - bias5first ;

subplot(1,3,3) ;
loglog(var1first,'linewidth',2) ;
hold on
loglog(var2first,'linewidth',2) ;
loglog(var3first,'linewidth',2) ;
loglog(var4first,'linewidth',2) ;
loglog(var5first,'linewidth',2) ; grid on
set(gca,'fontsize',24) ;
set(gca,'TickLabelInterpreter','latex') ;
xlabel('$K$','interpreter','latex') ;
ylabel('Variance','interpreter','latex') ;
hold off



figure(2) ;
subplot(1,3,1) ;
loglog(mse1sec,'linewidth',2) ;
hold on
loglog(mse2sec,'linewidth',2) ;
loglog(mse3sec,'linewidth',2) ;
loglog(mse4sec,'linewidth',2) ;
loglog(mse5sec,'linewidth',2) ; grid on
set(gca,'fontsize',24) ;
set(gca,'TickLabelInterpreter','latex') ;
xlabel('$K$','interpreter','latex') ;
ylabel('MSE','interpreter','latex') ;
hold off ; 
legend({'LD','RM','Irr','RMIrr','GiIrr'},'interpreter','latex') ;


subplot(1,3,2) ;
loglog(bias1sec,'linewidth',2) ;
hold on
loglog(bias2sec,'linewidth',2) ;
loglog(bias3sec,'linewidth',2) ;
loglog(bias4sec,'linewidth',2) ;
loglog(bias5sec,'linewidth',2) ; grid on
set(gca,'fontsize',24) ;
set(gca,'TickLabelInterpreter','latex') ;
xlabel('$K$','interpreter','latex') ;
ylabel('Bias$^2$','interpreter','latex') ;
title('$\phi(\theta) = \theta_1^2+\theta_2^2+\theta_3^2$','interpreter','latex'); 
hold off  


var1sec = mse1sec - bias1sec ;
var2sec = mse2sec - bias2sec ;
var3sec = mse3sec - bias3sec ;
var4sec = mse4sec - bias4sec ;
var5sec = mse5sec - bias5sec ;

subplot(1,3,3) ;
loglog(var1sec,'linewidth',2) ;
hold on
loglog(var2sec,'linewidth',2) ;
loglog(var3sec,'linewidth',2) ;
loglog(var4sec,'linewidth',2) ;
loglog(var5sec,'linewidth',2) ; grid on
set(gca,'fontsize',24) ;
set(gca,'TickLabelInterpreter','latex') ;
xlabel('$K$','interpreter','latex') ;
ylabel('Variance','interpreter','latex') ;
hold off


%% Asymptotic variances
burnin = 5/h ; 
 Avarfirstmom1 = zeros(M,1) ; Avarsecmom1 = zeros(M,1) ;
 Avarfirstmom2 = zeros(M,1) ; Avarsecmom2 = zeros(M,1) ;
Avarfirstmom3 = zeros(M,1) ; Avarsecmom3 = zeros(M,1) ;
Avarfirstmom4 = zeros(M,1) ; Avarsecmom4 = zeros(M,1) ;
Avarfirstmom5 = zeros(M,1) ; Avarsecmom5 = zeros(M,1) ;

K = Nsteps; 
Kint = round((K-burnin)/20) ;
burnin = burnin+1; 
for ii = 1:M
%     obsmu1 = Yall(1,burnin+1:end,ii) ; obsmu1 = reshape(obsmu1,Kint,20) ; Avarmu1(ii) = var(mean(obsmu1)) ;
%     obsmu2 = Yrmall(1,burnin+1:end,ii) ; obsmu2 = reshape(obsmu2,Kint,20) ;Avarmu2(ii) = var(mean(obsmu2)) ;
%     obsmu3 = Yirall(1,burnin+1:end,ii) ; obsmu3 = reshape(obsmu3,Kint,20) ; Avarmu3(ii) = var(mean(obsmu3)) ;
%     obsmu4 = Yirmall(1,burnin+1:end,ii) ; obsmu4 = reshape(obsmu4,Kint,20) ;Avarmu4(ii) = var(mean(obsmu4)) ;
%     obsmu5 = Yirrmall(1,burnin+1:end,ii) ; obsmu5 = reshape(obsmu5,Kint,20) ;Avarmu5(ii) = var(mean(obsmu5)) ;
% 
%     obss1 = Yall(2,burnin+1:end,ii) ; obss1 = reshape(obss1,Kint,20) ; Avars1(ii) = var(mean(obss1)) ; 
%     obss2 = Yrmall(2,burnin+1:end,ii) ; obss2 = reshape(obss2,Kint,20) ; Avars2(ii) = var(mean(obss2)) ; 
%     obss3 = Yirall(2,burnin+1:end,ii) ; obss3 = reshape(obss3,Kint,20) ; Avars3(ii) = var(mean(obss3)) ; 
%     obss4 = Yirmall(2,burnin+1:end,ii) ; obss4 = reshape(obss4,Kint,20) ; Avars4(ii) = var(mean(obss4)) ; 
%     obss5 = Yirrmall(2,burnin+1:end,ii) ; obss5 = reshape(obss5,Kint,20) ; Avars5(ii) = var(mean(obss5)) ; 

    obsfirst1 = sum(t1(:,burnin+1:end,ii)) ; obsfirst1 = reshape(obsfirst1,Kint,20) ; Avarfirstmom1(ii) = var(mean(obsfirst1)) ;
    obsfirst2 = sum(t2(:,burnin+1:end,ii)); obsfirst2 = reshape(obsfirst2,Kint,20) ; Avarfirstmom2(ii) = var(mean(obsfirst2)) ;
    obsfirst3 = sum(t3(:,burnin+1:end,ii)) ; obsfirst3 = reshape(obsfirst3,Kint,20) ; Avarfirstmom3(ii) = var(mean(obsfirst3)) ;
    obsfirst4 = sum(t4(:,burnin+1:end,ii)) ; obsfirst4 = reshape(obsfirst4,Kint,20) ; Avarfirstmom4(ii) = var(mean(obsfirst4)) ;
    obsfirst5 = sum(t5(:,burnin+1:end,ii)) ; obsfirst5 = reshape(obsfirst5,Kint,20) ; Avarfirstmom5(ii) = var(mean(obsfirst5)) ;

    
    obssec1 = sum(t1(:,burnin+1:end,ii).^2) ; obssec1 = reshape(obssec1,Kint,20) ; Avarsecmom1(ii) = var(mean(obssec1)) ;
    obssec2 = sum(t2(:,burnin+1:end,ii).^2); obssec2 = reshape(obssec2,Kint,20) ; Avarsecmom2(ii) = var(mean(obssec2)) ;
    obssec3 = sum(t3(:,burnin+1:end,ii).^2) ; obssec3 = reshape(obssec3,Kint,20) ; Avarsecmom3(ii) = var(mean(obssec3)) ;
    obssec4 = sum(t4(:,burnin+1:end,ii).^2) ; obssec4 = reshape(obssec4,Kint,20) ; Avarsecmom4(ii) = var(mean(obssec4)) ;
    obssec5 = sum(t5(:,burnin+1:end,ii).^2) ; obssec5 = reshape(obssec5,Kint,20) ; Avarsecmom5(ii) = var(mean(obssec5)) ;

    
    
    
end


% 



% 
