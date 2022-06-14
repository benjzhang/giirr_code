function dlogpi = gradlogpos_irrwriem_new(W,X,J,lambda)

d = size(X,1) ;
N = size(X,2) ;
n = round(0.1*N) ;

summat1 = zeros(d) ;
summat2 = zeros(d); 
W = reshape(W,d,d) ;

index = randsample(N,n) ;
for ii = 1:size(index)
    
    xn = X(:,index(ii)) ;
    yn = W * xn ;
    t = tanh(0.5 * yn) ;
    summat1 = summat1 + t * yn'  ;
    summat2 = summat2 + t * xn' ;
end

dlogpi1 = ( N * eye(d) -  N/n * summat1) * W - lambda * W * (W'*W) +(d+1) * W ;
dlogpi2 = (N *eye(d) / (W') - N/n * summat2) - lambda * W ;

dlogpi = dlogpi1(:) + dlogpi2(:) + J * dlogpi2(:);


end

