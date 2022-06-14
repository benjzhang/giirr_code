function output = gradlogpos_blr_irriem(w,alpha,X,t,n,G,J)



output = -alpha * w + loglikelihood_blr(w,t,X,n); 
output = output + 2*J * output + (eye(length(w)) + J) * (G\output) + G\(J*output) - correction2(X,w,G,J) ;



end