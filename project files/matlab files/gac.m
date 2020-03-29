function[phi, im] = gac(im, dt, c, iter_max, fignum, phi_type, noisy)
%% Matlab code modified from 
%     http://www.math.ucla.edu/~lvese/285j.1.05s/TV_L2.m
%
% Image segmentation by geodesic active contours (GAC) model as described
% in [1] via a semi-implicit discretization similar to [2]. The energy to 
% be minimized is 
%     integrate g( grad(I) ) ds
% and its Euler-Lagrange equation 
%     u_t = agrad(u) * div( g*grad(u)/agrad(u) ) + c*g*agrad(u)
%         = g*(c + kappa)*agrad(u) + <grad(u), grad(g)>,
% where agrad(u) = abs(grad(u)). Upwinding is applied to the 2nd term [3].
%
% Inputs:
%   im: select synthetic image (see options below) for segmentation
%   c: constant velocity in the normal dir; c>0 (expand), c<0 (contract)
%   iter_max: max # of iterations
%   fignum: figure# for plotting
%   phi_type: level set function initialization
%   dt: time stepsize
%   noisy: 1 or 0
% 
% Output:
%   phi: level set function
%
% Eg:
% >> gac('sqr4', 0.05, -9, 300);
%
% References:
% [1] CassellKimmelSapiro, Geodesic active contours (1997)
% [2] ChanVese, Active contours without edges (2001)
% [3] MarquinaOsher, Explicit algorithms for a new time-dependent...(2000)
%
% LAST MODIFIED: 25Mar2020
%
 

%% Read inputs
  if nargin < 1
    error('Missing inputs');
  elseif nargin < 7
    noisy = 0;
    if nargin < 6
      phi_type = im;
      if nargin < 5
        fignum = 96;
        if nargin < 4
          iter_max = 100;
          if nargin < 3
            c = 0;
            if nargin < 2
              dt = 0.05;
            end
          end
        end
      end
    end
  end
      

%% Load initial data
%   u0 = load('noisybrain.mat'); u0 = 255*im2double(u0.Kn);

% Create simple synthetic images for testing
  u0 = zeros(256,256);
  switch im
    case 'sqr1' % centred square
      u0(100:150,100:150) = 255; 
      r = 45;
      
    case 'sqr2' % offset squares
      u0(50:100,50:100) = 255; 
      u0(150:170,150:170) = 255;
      
    case 'sqr3' % L-shape
      u0(78:178,78:128) = 255;
      u0(129:178,129:179) = 255; 
      r = 75;
      
    case 'sqr4' % L-shape + small block
      u0(78:178, 78:128) = 255;
      u0(129:178, 129:179) = 255; 
      u0(78:118, 139:179) = 255;
      r = 85;
      
    case 'blur1' % gaussian bump; blurred circle
      xx = 1:256;
      [XX, YY] = meshgrid(xx, xx);
      D = sqrt((XX-128).^2 + (YY-128).^2);
      u0( D < 25 ) = 256;
      u0 = imgaussfilt( u0, 15 ); 
      r = 125;
      
    case 'blur2' % gaussian bump; blurred circle
      xx = 1:256;
      [XX, YY] = meshgrid(xx, xx);
      D = sqrt((XX-128).^2 + (YY-128).^2);
      u0( D < 25 ) = 255;
      u0 = imgaussfilt( u0, 40 ); 
      r = 125;
      
    case 'sidebar'
      u0(96:160, 112:128) = 255;
      for i = 129:192
        u0(96:160, i) = -1/16*(i-128).^2 + 255;
      end
      u0 = circshift(u0, [0,-111]);
      r = 111;
      
    case 'bar2'
      u0(96:160, 64:128) = 255;
      epsl = 1e-1;
      p = 2;
      a = (epsl - 4^4)/(4^(3/p));
      for i = 129:192
        u0(96:160, i) = a*(i-128)^(1/p) + 255;
      end
      r = 75;
      
  end
      
  [M, N] = size(u0);
  
  % Add noise
  if noisy == 1
    u0 = u0 + 10*randn(size(u0));
    for i = 1:3
      u0 = imgaussfilt(u0,3);
    end
  end

% Show initial image
  figure(fignum); clf; subplot(3,3,1)
  imagesc(u0); axis('image', 'off')
  title('\bf (Noisy) image - original', 'fontsize', 20);
  
  
%% Set parameters
% Space & time discretization 
  h = 1.0;
%   dt = 0.001;

% Regularize TV at the origin 
  ep = 1e-6;
  ep2 = ep*ep;
  
% Stopping tol
  tol = 1e-3;
  n1 = -1; n2 = -1;
  
  
%% Setup edge detector
%   g = imfilter( u0, fspecial('average', 3), 'replicate' ); % smoothed image
  g = imgaussfilt( u0 );   % gaussian-smoothed image
  g = 1./(1 + imgradient( g ).^2);  % classic edge detector, p=2
  
  
%% Initialize level set function
  x = linspace(1, N, N); 
  y = linspace(1, M, M);
  [X, Y] = meshgrid(x, y);
  
  % Option 1: one circle 
  switch phi_type
    case {'sqr1', 'sqr3', 'sqr4', 'blur1', 'blur2', 'bar2' }
%       r = 75; %min(M,N)/4;
      phi = -sqrt( (X-(N+1)/2).^2 + (Y-(M+1)/2).^2 ) + r;
      
    case 'sqr2'
      phi = -sqrt( (X-75).^2 + (Y-75).^2 ) + 40;
      phi = max(phi, -sqrt( (X-160).^2 + (Y-160).^2 ) + 20);
      
    case 'sidebar'
      phi = -sqrt( (X-(0)/2).^2 + (Y-(M+1)/2).^2 ) + r;
  end
  
%   phi = phi/max(abs(phi(:)));
    
  
%% %%%%%%%%%%    Begin iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for Iter=1:iter_max
  
%   fprintf('Iter = %3d, C1 = %8.9g, C2 = %3.8g\n', Iter);
  
% Update pointwise (Gauss-Seidel type scheme)  
  for i = 2:M-1
    for j = 2:N-1
      phix = ( phi(i+1,j) - phi(i,j) )/h;
      phiy = ( phi(i,j+1) - phi(i,j-1) )/(2*h);
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co1 = g(i,j) ./ Gradu;

      phix = ( phi(i,j) - phi(i-1,j) )/h;
      phiy = ( phi(i-1,j+1) - phi(i-1,j-1) )/(2*h);
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co2 = g(i-1,j) ./ Gradu;

      phix = ( phi(i+1,j) - phi(i-1,j) )/(2*h);
      phiy = ( phi(i,j+1) - phi(i,j) )/h;
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co3 = g(i,j) ./ Gradu;

      phix = ( phi(i+1,j-1) - phi(i-1,j-1) )/(2*h);
      phiy = ( phi(i,j) - phi(i,j-1) )/h;
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co4 = g(i,j-1) ./ Gradu;
      
      phix = ( phi(i+1,j) - phi(i-1,j) ) / (2*h);
      phiy = ( phi(i,j+1) - phi(i,j-1) ) / (2*h);
      agrad = sqrt( phix*phix + phiy*phiy + ep2 );
      co = 1.0 + dt*agrad*( co1+co2+co3+co4 );
      
    	div = co1*phi(i+1,j) + co2*phi(i-1,j) + co3*phi(i,j+1) ...
        + co4*phi(i,j-1);
      
      gx = -c*(phi(i+1,j) - phi(i-1,j));
      gy = -c*(phi(i,j+1) - phi(i,j-1));
      if gx > 0
        gx = (phi(i,j) - phi(i-1,j))/h;
      else
        gx = (phi(i+1,j) - phi(i,j))/h;
      end
      if gy > 0
        gy = (phi(i,j) - phi(i,j-1))/h;
      else
        gy = (phi(i,j+1) - phi(i,j))/h;
      end

      phi(i,j) = (1./co) * ( phi(i,j) + dt*agrad*div + ...
        dt*c*g(i,j)*sqrt( gx.^2 + gy.^2 ) );

    end
  end
% End pointwise updates

% Update boundaries
  phi = BCs(phi, M, N);
 
% Stopping criteria
  n1 = nnz(phi<0);
  if Iter > 10 && abs(n1 - n2) < 2
    break;
  else
    n2 = n1;
  end

% Mid-cycle plot updates
  if mod(Iter, iter_max/10) == 1    % change 500 to small# for more updates
    plotseg(u0, phi, fignum, c, Iter, phi_type);
    pause(0.05)
  end
  
end 
% %%%%%%%%%%     End iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Plot final results
  plotseg(u0, phi, fignum, c, Iter, phi_type);
  fprintf('\n');
end
% End of main function

function[] = plotseg(u0, phi, fignum, c, Iter, phi_type)
%% Visualize final results 
%
  figure(fignum); subplot(3,3,[2 3 5 6 8 9]);
  imagesc(u0);  axis('image', 'off'); colormap(gray); 
  hold on 
  contour(phi, [0,0], 'linewidth', 1.5, 'linecolor', 'r');
  hold off;
  
  title({['\bf GAC -- ', phi_type], ... 
    ['c = ', num2str(c), ' , Iter = ', num2str(Iter)]}, ...
    'fontsize', 20)
  
  subplot(3,3,[4 7])
  contourf(flipud(phi), [0 0], 'linewidth', 1.5);
  axis('image', 'off'); colormap(gray);
  title('\bf Level set function' ,'fontsize', 20)
  
  h = gca; 
  h.FontSize = 18;
  h.TickLabelInterpreter = 'latex';
  
end

function[phi] = BCs(phi, M, N)
%% Sets homogeneous neumann BCs
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