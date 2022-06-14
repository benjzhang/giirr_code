function output = correction(X,w,G) 
d = size(w,1); 
logiteval = logisticfunc(X'*w); 
Lambda = diag(logiteval .*(1-logiteval)) ;
output = zeros(d,1); 

% G = X * Lambda *X' + alpha * eye(d) ;

for ii = 1:d
    Vii = diag((1-2*logisticfunc(X'*w)) .* X(ii,:)') ;
    dG =  X * Lambda * Vii * X' ;
    termii = (G\(dG))/G ;
    output = output + termii(:,ii); 
    
end







end