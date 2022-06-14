syms w1 w2
d = 2; 
X = xtrain'; 
alpha = 1; 


% vars = [w11 w21 w31 w12 w22 w32 w13 w23 w33] ;
% W = [w11 w12; w21 w22]; 
% W = [w11 w12 w13 w14; w21 w22 w23  w24; w31 w32 w33 w34; w41 w42 w43 w44  ]; 
w = [w1; w2] ; 
% B = inv(metric(X,w,alpha)); 
B = metric(X,w,alpha) \ eye(d); 

sym cor
for ii = 1:d
   
    cor(ii,:) = divergence(B(ii,:),w); 
    
end

% matlabFunction(cor,'File','blrdiv','Vars',{W(:)});
