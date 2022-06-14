function gradl = loglikelihood_blr(w,t,X,n)

N = size(X,2) ;
if n < N
    index = randsample(N,n) ;
    Xnow = X(:,index) ;
    tnow = t(index) ;
else
    Xnow = X ;
    tnow = t ;
end

phi = @(y) 1./(1+ exp(-y)) ;
gradl = N/n * Xnow * tnow - N/n * (Xnow * phi(Xnow'* w)) ;










end