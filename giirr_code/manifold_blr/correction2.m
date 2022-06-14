function output = correction2(X,w,G,J) 
d = size(w,1); 
logiteval = logisticfunc(X'*w); 
Lambda = diag(logiteval .*(1-logiteval)) ;
output = zeros(d,1); 

% G = X * Lambda *X' + alpha * eye(d) ;

for ii = 1:d
    Vii = diag((1-2*logisticfunc(X'*w)) .* X(ii,:)') ;
    dG =  X * Lambda * Vii * X' ;
    termii = (G\(dG))/G ;
    termii = termii + J * termii + termii * J; 
    output = output + termii(:,ii); 
    
end

% output = output + J * output + output * J ;







end