function[phi, x, y] = init_ls( N, M, r, phi_type )
%% Initialize level set function for gac.m and acwe.m
  x = linspace(1, N, N); 
  y = linspace(1, M, M);
  [X, Y] = meshgrid(x, y);
  
  switch phi_type
    case {'sqr1', 'sqr3', 'sqr4', 'blur', 'blur2', 'bar' }
      phi = -sqrt( (X-(N+1)/2).^2 + (Y-(M+1)/2).^2 ) + r;
      
    case 'sqr2'
      phi = -sqrt( (X-75).^2 + (Y-75).^2 ) + 40;
      phi = max(phi, -sqrt( (X-160).^2 + (Y-160).^2 ) + 20);
      
    case 'sidebar'
      phi = -sqrt( (X-(0)/2).^2 + (Y-(M+1)/2).^2 ) + r;
      
    case 'circle'
      r = min(M,N)/4;
      phi = -sqrt( (X-(N+1)/2).^2 + (Y-(M+1)/2).^2 ) + r;
  
    case 'bubbles'
      r = min( min(M,N)/4, 7 );
      phi = -sqrt(M^2 + N^2)*ones(size(X));
      for i = 1:floor(N/25)
        for j = 1:floor(M/25)
          phi = max(phi, -sqrt( (X- 25*(i-1)-12.5).^2 + (Y-25*(j-1)-12.5).^2 ) + r);
        end
      end  
      
    case {'grid'}
      phi = -sqrt( (X-(N+1)/2).^2 + (Y-(M+1)/2).^2 ) + 30;
      phi = max(phi, -sqrt( (X-(N+1)/2).^2 + (Y-190).^2 ) + 25);
      phi = max(phi, -sqrt( (X -(132)/2).^2 + (Y-125/2).^2 ) + 27);
%       phi = -sqrt( (X-(N+1)/2).^2 + (Y-(M+1)/2).^2 ) + r;
%       
    case 'blank'
      phi = -sqrt( (X-(N+1)/2).^2 + (Y-(M+1)/2).^2 ) + r;
  end
end