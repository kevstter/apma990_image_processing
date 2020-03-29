function[phi] = BCs(phi, M, N)
%% Sets homogeneous neumann BCs for image segmentation codes
%
  for i = 2:M-1
    phi(i,1) = phi(i,2);
    phi(i,N) = phi(i,N-1);
  end

	for j = 2:N-1
    phi(1,j) = phi(2,j);
    phi(M,j) = phi(M-1,j);
  end

  phi(1,1) = phi(2,2);
  phi(1,N) = phi(2,N-1); 
  phi(M,1) = phi(M-1,2);
  phi(M,N) = phi(M-1,N-1);
  
end