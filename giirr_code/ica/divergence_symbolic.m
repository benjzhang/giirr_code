syms w11 w21 w31 w12 w22 w32 w13 w23 w33 w41 w42 w43 w44 w14 w24 w34

% vars = [w11 w21 w31 w12 w22 w32 w13 w23 w33] ;
% W = [w11 w12; w21 w22]; 
W = [w11 w12 w13 ; w21 w22 w23 ; w31 w32 w33 ]; 
% W = [w11 w12 w13 w14; w21 w22 w23  w24; w31 w32 w33 w34; w41 w42 w43 w44  ]; 

% C = const * C00 ;

% field = 2 * kron(W.'*W, C)+ kron(C * (W.'*W),eye(d)) + kron(W.'*W*C,eye(d)) ; 
field = J * kron(W.'*W,eye(d)) + kron(W.'*W,eye(d)) * J ;
%  field = 0.5 * J * kron(W.'*W,eye(d)) + 0.5 * kron(W.'*W,eye(d)) * J ;

sym cor
for ii = 1:d^2
   
    cor(ii,:) = divergence(field(ii,:),W(:)); 
    
end

matlabFunction(cor,'File','irrev_correc','Vars',{W(:)});
