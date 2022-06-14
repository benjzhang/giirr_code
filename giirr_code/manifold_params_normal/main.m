% rng(12847821) ;
rng(12847821) ;

%% Setup
% Data
N = 30;
mutrue = 0;
sigmatrue = 10 ;
X = mutrue + sigmatrue * randn(N,1) ;

% Discretization
M = 100;
h = 0.001;
T = 1000 ;
K = T/h + 1 ;
initcond = [5;20] ;

rng(283482) ;
for ll = 1:10

%% Standard Langevin
Yall = zeros(2,K,M) ;

parfor mm = 1:M
    Y = zeros(2,K) ;
    Y(:,1) = initcond;
    for kk = 1:K-1
        Y(:,kk+1) = Y(:,kk) + h/2 * gradfunc(Y(:,kk), X) + sqrt(h) * randn(2,1) ;
    end
    Yall(:,:,mm) = Y ;
end


% RM Langevin
B = @(state) state(2).^2/N * diag([1,1/2]) ;
divB = @(state) state(2) / N *[0 ; 1];
Yrmall = zeros(2,K,M) ;

parfor mm = 1:M
    Yrm = zeros(2,K) ;
    Yrm(:,1) = initcond ;
    
    for kk = 1:K-1
        Beval = B(Yrm(:,kk)) ;
        divBeval = divB(Yrm(:,kk)) ;
        gradeval = gradfunc(Yrm(:,kk),X) ;
        
        Yrm(:,kk+1) = Yrm(:,kk) + h/2 * (Beval * gradeval + divBeval) +...
            sqrt(h * Beval) * randn(2,1) ;
    end
    Yrmall(:,:,mm) = Yrm ;
end


%% Irreversible Langevin
delta = 2;
J = delta * [0 1; -1 0] ;

Yirall = zeros(2,K,M) ;

parfor mm = 1:M
    Yir = zeros(2,K) ;
    Yir(:,1) = initcond ;
    
    for kk = 1:K-1
        gradeval = gradfunc(Yir(:,kk),X) ;
        
        Yir(:,kk+1) = Yir(:,kk) + h * (1/2 * eye(2) + J) * gradeval  +...
            sqrt(h) * randn(2,1) ;
    end
    Yirall(:,:,mm) = Yir ;
end

%% Irreversible on RM Langevin -- irreversibility applied without knowledge of RM

Yirmall = zeros(2,K,M) ;

parfor mm = 1:M
    Yirm = zeros(2,K) ;
    Yirm(:,1) = initcond ;
    
    for kk = 1:K-1
        Beval = B(Yirm(:,kk)) ;
        divBeval = divB(Yirm(:,kk)) ;
        gradeval = gradfunc(Yirm(:,kk),X) ;
        
        Yirm(:,kk+1) = Yirm(:,kk) + h * ((1/2*Beval+J) * gradeval + 1/2*divBeval) +...
            sqrt(h * Beval) * randn(2,1) ;
    end
    Yirmall(:,:,mm) = Yirm ;
end


%% Irreversible RM Langevin

B = @(state) state(2).^2/N * diag([1,1/2]) ;
divB = @(state) state(2) / N *[0 ; 1];

delta = 2;

J = delta * [0 1; -1 0] ;
C = @(state) 0.75 * state(2)^2/N * J ;
divC = @(state) 3/2 * delta * state(2) /N * [1; 0] ;


Yirrmall = zeros(2,K,M) ;

parfor mm = 1:M
    Yirrm = zeros(2,K) ;
    Yirrm(:,1) = initcond ;
    
    for kk = 1:K-1
        Beval = B(Yirrm(:,kk)) ; Ceval = C(Yirrm(:,kk)) ;
        divBeval = divB(Yirrm(:,kk)) ; divCeval = divC(Yirrm(:,kk)) ;
        gradeval = gradfunc(Yirrm(:,kk),X) ;
        
        Yirrm(:,kk+1) = Yirrm(:,kk) + h * ( (1/2 * Beval+Ceval) * gradeval + 1/2*divBeval + divCeval) +...
            sqrt(h * Beval) * randn(2,1) ;
    end
    Yirrmall(:,:,mm) = Yirrm ;
end

%% Estimators

mupost = 0.497820258605897 ; Exmu2 = 3.538605250460973 ;
sigmapost = 9.840908347941536 ; Exsigma2 =   98.723407211684645 ;
burnin = 50/h ; 

estimatorfirst1 = zeros(K-burnin,M) ; estimatorsec1 = zeros(K-burnin,M) ;
estimatorfirst2 = zeros(K-burnin,M) ; estimatorsec2 = zeros(K-burnin,M) ;
estimatorfirst3 = zeros(K-burnin,M) ; estimatorsec3 = zeros(K-burnin,M) ;
estimatorfirst4 = zeros(K-burnin,M) ; estimatorsec4 = zeros(K-burnin,M) ;
estimatorfirst5 = zeros(K-burnin,M) ; estimatorsec5 = zeros(K-burnin,M) ;

for ii = 1:M
   
%     estimator1(:,:,ii) = cumsum(Yall(:,burnin+1:end,ii),2)./repmat((1:K-burnin),[2,1]) ;
%     estimator2(:,:,ii) = cumsum(Yrmall(:,burnin+1:end,ii),2)./repmat((1:K-burnin),[2,1]) ;
%     estimator3(:,:,ii) = cumsum(Yirall(:,burnin+1:end,ii),2)./repmat((1:K-burnin),[2,1]) ;
%     estimator4(:,:,ii) = cumsum(Yirmall(:,burnin+1:end,ii),2)./repmat((1:K-burnin),[2,1]) ;
%     estimator5(:,:,ii) = cumsum(Yirrmall(:,burnin+1:end,ii),2)./repmat((1:K-burnin),[2,1]) ;

     estimatorfirst1(:,ii) = cumsum(sum(Yall(:,burnin+1:end,ii)),2)./(1:K-burnin) ;
    estimatorfirst2(:,ii) = cumsum(sum(Yrmall(:,burnin+1:end,ii)),2)./(1:K-burnin) ;
    estimatorfirst3(:,ii) = cumsum(sum(Yirall(:,burnin+1:end,ii)),2)./(1:K-burnin) ;
    estimatorfirst4(:,ii) = cumsum(sum(Yirmall(:,burnin+1:end,ii)),2)./(1:K-burnin) ;
    estimatorfirst5(:,ii) = cumsum(sum(Yirrmall(:,burnin+1:end,ii)),2)./(1:K-burnin) ;

    estimatorsec1(:,ii) = cumsum(sum(Yall(:,burnin+1:end,ii).^2),2)./(1:K-burnin) ;
    estimatorsec2(:,ii) = cumsum(sum(Yrmall(:,burnin+1:end,ii).^2),2)./(1:K-burnin) ;
    estimatorsec3(:,ii) = cumsum(sum(Yirall(:,burnin+1:end,ii).^2),2)./(1:K-burnin) ;
    estimatorsec4(:,ii) = cumsum(sum(Yirmall(:,burnin+1:end,ii).^2),2)./(1:K-burnin) ;
    estimatorsec5(:,ii) = cumsum(sum(Yirrmall(:,burnin+1:end,ii).^2),2)./(1:K-burnin) ;

end

% mse1mu = mean((estimator1(1,:,:) - mupost).^2,3) ; bias1musq = mean(estimator1(1,:,:)-mupost,3).^2 ;
% mse2mu = mean((estimator2(1,:,:) - mupost).^2,3) ; bias2musq = mean(estimator2(1,:,:)-mupost,3).^2 ;
% mse3mu = mean((estimator3(1,:,:) - mupost).^2,3) ; bias3musq = mean(estimator3(1,:,:)-mupost,3).^2 ;
% mse4mu = mean((estimator4(1,:,:) - mupost).^2,3) ; bias4musq = mean(estimator4(1,:,:)-mupost,3).^2 ;
% mse5mu = mean((estimator5(1,:,:) - mupost).^2,3) ; bias5musq = mean(estimator5(1,:,:)-mupost,3).^2 ;
% 
% var1mu = mse1mu - bias1musq ;
% var2mu = mse2mu - bias2musq ;
% var3mu = mse3mu - bias3musq ;
% var4mu = mse4mu - bias4musq ;
% var5mu = mse5mu - bias5musq ;
% 
% 
% mse1s = mean((estimator1(2,:,:) - sigmapost).^2,3) ; bias1ssq = mean(estimator1(2,:,:)-sigmapost,3).^2 ;
% mse2s = mean((estimator2(2,:,:) - sigmapost).^2,3) ; bias2ssq = mean(estimator2(2,:,:)-sigmapost,3).^2 ;
% mse3s = mean((estimator3(2,:,:) - sigmapost).^2,3) ; bias3ssq = mean(estimator3(2,:,:)-sigmapost,3).^2 ;
% mse4s = mean((estimator4(2,:,:) - sigmapost).^2,3) ; bias4ssq = mean(estimator4(2,:,:)-sigmapost,3).^2 ;
% mse5s = mean((estimator5(2,:,:) - sigmapost).^2,3) ; bias5ssq = mean(estimator5(2,:,:)-sigmapost,3).^2 ;
% 
% var1s = mse1s - bias1ssq ;
% var2s = mse2s - bias2ssq ;
% var3s = mse3s - bias3ssq ;
% var4s = mse4s - bias4ssq ;
% var5s = mse5s - bias5ssq ;
% 
% 

mse1first = mean((estimatorfirst1 - mupost-sigmapost).^2,2) ; bias1firstsq = mean(estimatorfirst1-mupost-sigmapost,2) ;
mse2first = mean((estimatorfirst2 - mupost-sigmapost).^2,2) ; bias2firstsq = mean(estimatorfirst2-mupost-sigmapost,2) ;
mse3first = mean((estimatorfirst3 - mupost-sigmapost).^2,2) ; bias3firstsq = mean(estimatorfirst3-mupost-sigmapost,2) ;
mse4first = mean((estimatorfirst4 - mupost-sigmapost).^2,2) ; bias4firstsq = mean(estimatorfirst4-mupost-sigmapost,2) ;
mse5first = mean((estimatorfirst5 - mupost-sigmapost).^2,2) ; bias5firstsq = mean(estimatorfirst5-mupost-sigmapost,2) ;

% var1first = mse1first - bias1firstsq ;
% var2first = mse2first - bias2firstsq ;
% var3first = mse3first - bias3firstsq ;
% var4first = mse4first - bias4firstsq ;
% var5first = mse5first - bias5firstsq ;


mse1sec = mean((estimatorsec1 - Exmu2-Exsigma2).^2,2) ; bias1secsq = mean(estimatorsec1-Exmu2-Exsigma2,2) ;
mse2sec = mean((estimatorsec2 - Exmu2-Exsigma2).^2,2) ; bias2secsq = mean(estimatorsec2-Exmu2-Exsigma2,2) ;
mse3sec = mean((estimatorsec3 - Exmu2-Exsigma2).^2,2) ; bias3secsq = mean(estimatorsec3-Exmu2-Exsigma2,2) ;
mse4sec = mean((estimatorsec4 - Exmu2-Exsigma2).^2,2) ; bias4secsq = mean(estimatorsec4-Exmu2-Exsigma2,2) ;
mse5sec = mean((estimatorsec5 - Exmu2-Exsigma2).^2,2) ; bias5secsq = mean(estimatorsec5-Exmu2-Exsigma2,2) ;

% var1sec = mse1sec - bias1secsq ;
% var2sec = mse2sec - bias2secsq ;
% var3sec = mse3sec - bias3secsq ;
% var4sec = mse4sec - bias4secsq ;
% var5sec = mse5sec - bias5secsq ;

%% Asymptotic variances
burnin = 50/h ; 
 Avarfirstmom1 = zeros(M,1) ; Avarsecmom1 = zeros(M,1) ;
 Avarfirstmom2 = zeros(M,1) ; Avarsecmom2 = zeros(M,1) ;
Avarfirstmom3 = zeros(M,1) ; Avarsecmom3 = zeros(M,1) ;
Avarfirstmom4 = zeros(M,1) ; Avarsecmom4 = zeros(M,1) ;
Avarfirstmom5 = zeros(M,1) ; Avarsecmom5 = zeros(M,1) ;

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

    obsfirst1 = sum(Yall(:,burnin+1:end,ii)) ; obsfirst1 = reshape(obsfirst1,Kint,20) ; Avarfirstmom1(ii) = var(mean(obsfirst1)) ;
    obsfirst2 = sum(Yrmall(:,burnin+1:end,ii)); obsfirst2 = reshape(obsfirst2,Kint,20) ; Avarfirstmom2(ii) = var(mean(obsfirst2)) ;
    obsfirst3 = sum(Yirall(:,burnin+1:end,ii)) ; obsfirst3 = reshape(obsfirst3,Kint,20) ; Avarfirstmom3(ii) = var(mean(obsfirst3)) ;
    obsfirst4 = sum(Yirmall(:,burnin+1:end,ii)) ; obsfirst4 = reshape(obsfirst4,Kint,20) ; Avarfirstmom4(ii) = var(mean(obsfirst4)) ;
    obsfirst5 = sum(Yirrmall(:,burnin+1:end,ii)) ; obsfirst5 = reshape(obsfirst5,Kint,20) ; Avarfirstmom5(ii) = var(mean(obsfirst5)) ;

    
    obssec1 = sum(Yall(:,burnin+1:end,ii).^2) ; obssec1 = reshape(obssec1,Kint,20) ; Avarsecmom1(ii) = var(mean(obssec1)) ;
    obssec2 = sum(Yrmall(:,burnin+1:end,ii).^2); obssec2 = reshape(obssec2,Kint,20) ; Avarsecmom2(ii) = var(mean(obssec2)) ;
    obssec3 = sum(Yirall(:,burnin+1:end,ii).^2) ; obssec3 = reshape(obssec3,Kint,20) ; Avarsecmom3(ii) = var(mean(obssec3)) ;
    obssec4 = sum(Yirmall(:,burnin+1:end,ii).^2) ; obssec4 = reshape(obssec4,Kint,20) ; Avarsecmom4(ii) = var(mean(obssec4)) ;
    obssec5 = sum(Yirrmall(:,burnin+1:end,ii).^2) ; obssec5 = reshape(obssec5,Kint,20) ; Avarsecmom5(ii) = var(mean(obssec5)) ;

    
    
    
end

name = sprintf('runmoreburn1%i.mat',ll); 
save(name,'Avarfirstmom1','Avarfirstmom2','Avarfirstmom3','Avarfirstmom4','Avarfirstmom5','Avarsecmom1','Avarsecmom2','Avarsecmom3',...
    'Avarsecmom4','Avarsecmom5','mse1first','mse2first','mse3first','mse4first','mse5first','bias1firstsq','bias2firstsq','bias3firstsq',...
    'bias4firstsq','bias5firstsq','mse1sec','mse2sec','mse3sec','mse4sec','mse5sec','bias1secsq','bias2secsq','bias3secsq',...
    'bias4secsq','bias5secsq') ;



end



exit

%% Figures
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
loglog(bias1firstsq,'linewidth',2) ;
hold on
loglog(bias2firstsq,'linewidth',2) ;
loglog(bias3firstsq,'linewidth',2) ;
loglog(bias4firstsq,'linewidth',2) ;
loglog(bias5firstsq,'linewidth',2) ; grid on
set(gca,'fontsize',24) ;
set(gca,'TickLabelInterpreter','latex') ;
xlabel('$K$','interpreter','latex') ;
ylabel('Bias$^2$','interpreter','latex') ;
hold off  

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

% %
% figure(2) ;
% subplot(1,3,1) ;
% loglog(mse1s,'linewidth',2) ;
% hold on
% loglog(mse2s,'linewidth',2) ;
% loglog(mse3s,'linewidth',2) ;
% loglog(mse4s,'linewidth',2) ;
% loglog(mse5s,'linewidth',2) ; grid on
% set(gca,'fontsize',24) ;
% set(gca,'TickLabelInterpreter','latex') ;
% xlabel('Number of steps','interpreter','latex') ;
% ylabel('MSE','interpreter','latex') ;
% hold off ; 
% legend({'LD','RM','Irr','RM+Irr','RMIrr'},'interpreter','latex') ;
% 
% 
% subplot(1,3,2) ;
% loglog(bias1ssq,'linewidth',2) ;
% hold on
% loglog(bias2ssq,'linewidth',2) ;
% loglog(bias3ssq,'linewidth',2) ;
% loglog(bias4ssq,'linewidth',2) ;
% loglog(bias5ssq,'linewidth',2) ; grid on
% set(gca,'fontsize',24) ;
% set(gca,'TickLabelInterpreter','latex') ;
% xlabel('Number of steps','interpreter','latex') ;
% ylabel('Bias$^2$','interpreter','latex') ;
% hold off  
% 
% subplot(1,3,3) ;
% loglog(var1s,'linewidth',2) ;
% hold on
% loglog(var2s,'linewidth',2) ;
% loglog(var3s,'linewidth',2) ;
% loglog(var4s,'linewidth',2) ;
% loglog(var5s,'linewidth',2) ; grid on
% set(gca,'fontsize',24) ;
% set(gca,'TickLabelInterpreter','latex') ;
% xlabel('Number of steps','interpreter','latex') ;
% ylabel('Variance','interpreter','latex') ;
% hold off

%

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
xlabel('Number of steps','interpreter','latex') ;
ylabel('MSE','interpreter','latex') ;
hold off ; 
legend({'LD','RM','Irr','RMIrr','GiIrr'},'interpreter','latex') ;


subplot(1,3,2) ;
loglog(bias1secsq,'linewidth',2) ;
hold on
loglog(bias2secsq,'linewidth',2) ;
loglog(bias3secsq,'linewidth',2) ;
loglog(bias4secsq,'linewidth',2) ;
loglog(bias5secsq,'linewidth',2) ; grid on
set(gca,'fontsize',24) ;
set(gca,'TickLabelInterpreter','latex') ;
xlabel('Number of steps','interpreter','latex') ;
ylabel('Bias$^2$','interpreter','latex') ;
hold off  

subplot(1,3,3) ;
loglog(var1sec,'linewidth',2) ;
hold on
loglog(var2sec,'linewidth',2) ;
loglog(var3sec,'linewidth',2) ;
loglog(var4sec,'linewidth',2) ;
loglog(var5sec,'linewidth',2) ; grid on
set(gca,'fontsize',24) ;
set(gca,'TickLabelInterpreter','latex') ;
xlabel('Number of steps','interpreter','latex') ;
ylabel('Variance','interpreter','latex') ;
hold off



