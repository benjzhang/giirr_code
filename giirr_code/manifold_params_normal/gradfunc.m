function grad = gradfunc(state,data) 

%  n = 6 ;

mu = state(1) ;
sigma = state(2) ;


N = size(data,1) ;
% n = N; 
% index = randsample(N,n);
% data = data(index); 

m1 = sum(data - mu) ;
m2 =  sum((data-mu).^2) ;

grad = zeros(2,1) ;
grad(1) = m1 / sigma^2 ;
grad(2) = - N/sigma + m2 / sigma.^3 ; 


end
