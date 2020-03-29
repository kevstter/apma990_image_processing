function[vprime] = vp(u)
%% Derivative of a regularize version of the penalty function, v. 
% Useful for gcs.m
% See Figure 5 in [1].
  vep = 1e-4;
  vep2 = vep^2;
  
  if u <= -vep
    v = -2;
  elseif u > -vep && u < 0   
    % v = 1/(veps)*( u-veps );
    v = (2/vep2)*u^2*(u - 2*vep);    
  elseif u >= 0 && u <= 1
    v = 0;
  elseif u > 1 && u < 1+vep  
    % v = 1/(veps)*( u-1+eps );
    v = (-2/vep2)*(u-1)^2*(u-1-2*vep);
  else
    v = 2;
  end
  
  vprime = v;
end