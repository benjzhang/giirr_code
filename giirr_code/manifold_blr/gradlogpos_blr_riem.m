function output = gradlogpos_blr_riem(w,alpha,X,t,n,G)



output = -alpha * w + loglikelihood_blr(w,t,X,n); 

output = (G\output) + output - correction(X,w,G) ;



end