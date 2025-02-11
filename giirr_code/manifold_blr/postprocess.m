function [mse0first,mse0sec,bias0first,bias0sec] = postprocess(t0first,t0sec,secmom,mupost,burnIn)


Ntraj = size(t0first,3); 
Nsteps = size(t0first,2); 
% secmom = diag(inv(Gammap))+mupost.^2 ;

% burnIn = 5/h+1 ;
% d = size(t0,1); 
d = 1; 
est0first = zeros(1,Nsteps-burnIn+1,Ntraj) ;
est0sec = est0first; 

mse0first = zeros(1,Nsteps-burnIn+1,Ntraj) ;
mse0sec = mse0first ;


for ii = 1:Ntraj
    est0first(:,:,ii) = cumsum(t0first(:,burnIn:end,ii),2)./repmat(1:Nsteps-burnIn+1,d,1) ;
est0sec(:,:,ii) = cumsum(t0sec(:,burnIn:end,ii),2)./ repmat(1:Nsteps-burnIn+1,d,1);
est0firsttemp = sum(est0first(:,:,ii),1); 
est0sectemp = sum(est0sec(:,:,ii),1); 
% mse0(1,:,ii) = dot(est0(:,:,ii) - secmom,est0(:,:,ii) - secmom)  ;
mse0first(1,:,ii) = (est0firsttemp - sum(mupost)) .^2; 
mse0sec(1,:,ii) = (est0sectemp - sum(secmom)).^2 ;

end
mse0first = mean(mse0first,3) ;
mse0sec = mean(mse0sec,3); 

bias0first = (mean(sum(est0first,1),3)-sum(mupost)).^2 ;
bias0sec = (mean(sum(est0sec,1),3)-sum(secmom)).^2 ;








end

% function [mse0first,mse0sec,bias0first,bias0sec] = postprocess(t0,secmom,mupost,burnIn)
% 
% 
% Ntraj = size(t0,3); 
% Nsteps = size(t0,2); 
% % secmom = diag(inv(Gammap))+mupost.^2 ;
% 
% % burnIn = 5/h+1 ;
% d = size(t0,1); 
% est0first = zeros(d,Nsteps-burnIn+1,Ntraj) ;
% est0sec = est0first; 
% 
% mse0first = zeros(1,Nsteps-burnIn+1,Ntraj) ;
% mse0sec = mse0first ;
% 
% 
% for ii = 1:Ntraj
%     est0first(:,:,ii) = cumsum(t0(:,burnIn:end,ii),2)./repmat(1:Nsteps-burnIn+1,d,1) ;
% est0sec(:,:,ii) = cumsum(t0(:,burnIn:end,ii).^2,2)./ repmat(1:Nsteps-burnIn+1,d,1);
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
% 
% 
% 
% 
% 
% 
% end