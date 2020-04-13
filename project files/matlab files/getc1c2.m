function[C1,C2] = getc1c2(u,u0,thres)
% Compute region averages for fitting energy
%

  if nnz(u>thres) == 0
    C1 = 0;
  else
    C1 = mean(u0(u>thres),'all');
  end
  if nnz(u<=thres) == 0
    C2 = 0;
  else
    C2 = mean(u0(u<=thres),'all');
  end
  
end