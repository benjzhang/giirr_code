function output = gradlogpos_blr(w,alpha,X,t,n)



output = -alpha * w + loglikelihood_blr(w,t,X,n); 



end