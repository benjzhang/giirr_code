function output = gradlogpos_blr_irwriem(w,alpha,X,t,n,G,J)



output = -alpha * w + loglikelihood_blr(w,t,X,n); 

output = (G\output) + output - correction(X,w,G) + J * output ;



end