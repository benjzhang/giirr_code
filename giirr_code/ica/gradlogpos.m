function dlogpi = gradlogpos(W,X,lambda)

d = size(X,1) ;
N = size(X,2) ; 
n = round(0.1*N); 
summat = zeros(d) ;
W = reshape(W,d,d) ;

index = randsample(N,n); 
for ii = 1:length(index)
    
    xn = X(:,index(ii)) ;
    yn = W * xn ;
    summat = summat + tanh( 0.5 * yn) * xn' ;
end

dlogpi = (N * (eye(d)/(W')) -  N/n * summat) - lambda * W ;
dlogpi = dlogpi(:) ;

end

