function[u] = gcs(im, varargin)
%% gcs(im, lambda, edge, noisy, iter_max, fignum, u_type)
% Inputs:
%   im: select synthetic image (see options below) for segmentation
%   lambda: model parameter
%   edge: 0 or 1; 
%   noisy: 1 or 0
%   iter_max: max # of iterations
%   fignum: figure# for plotting
%   u_type: level set function initialization
% Output:
%   u: gradient descent solution; thresholding not yet applied
%
% Matlab code modified from 
%     http://www.math.ucla.edu/~lvese/285j.1.05s/TV_L2.m
%
% Image segmentation by the model introduced in [1] (convexification of 
% ACWE model [2] for 2-phase seg) and advanced in [3]. Gradient descent 
% solution via a semi-implicit discretization similar to [2]. The energy 
% to be minimized is 
%     integrate g*agrad(u) + integrate lambda*s(x) + alpha*v(u),
% and its Euler-Lagrange equation
%     u_t = div( g*grad(u)/agrad(u) ) + lambda*s(x) - alpha*v'(u)
% where agrad(u) = abs(grad(u)) and g is an edge indicator function.
%
% Eg:
% >> u = gcs('bar2', 20, 1, 1, 20, 90, 'same');
%
% References:
% [1] ChanEsedogluNikolova, Algos for finding global minimizers...(2006)
% [2] ChanVese, Active contours without edges (2001)
% [3] BressonEsedogluVanderheynstThiranOsher, Fast global minim...(2007)
%
% Created: 26Mar2020
% Last modified: 28Mar2020
%

%% Read inputs: (im, lambda, edge, noisy, iter_max, fignum, u_type)
  if nargin < 1
    error('Missing all inputs');
  end
  numvarargs = length(varargin);
  if numvarargs > 6
    error('Too many inputs...');
  end
  % lambda, edge, noisy, iter_max, fignum, u_type
  optargs = {1, 0, 0, 50, 90, 'default'};
  optargs(1:numvarargs) = varargin;
  [lambda, edge, noisy, iter_max, fignum, u_type] = optargs{:};

%% Set parameters
% Space & time discretization 
  h = 1.0;
  dt = 0.05;
  
% Model parameters
  alpha = lambda/2;
  thres = 0.50;
  oldC1 = -1; C1 = -1;                  % "initial" region avg: c1 and c2
  oldC2 = -1; C2 = -1;
  
% Regularize TV at the origin 
  ep = 1e-6;
  ep2 = ep*ep;
  
% Misc
  tol = 1e-6;                           % stopping tol  
  
%% Load initial data
%   u0 = load('noisybrain.mat'); u0 = 255*im2double(u0.Kn);

% Variety of simple synthetic images for testing
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
      
    case 'blur' % gaussian bump
      xx = 1:256;
      [XX, YY] = meshgrid(xx, xx);
      D = sqrt((XX-128).^2 + (YY-128).^2);
      u0( D < 25 ) = 256;
      u0 = imgaussfilt( u0, 15 ); 
      r = 125;
      
    case 'blur2' % gaussian bump
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
    
    otherwise
      fprintf('\nDefault u0 -- possibly undefined behaviour.\n');
      u0 = im2double(imread('cameraman.tif'));
%       u0 = im2double(rgb2gray(imread('pears.png')));
  end
  u0 = u0 / max(abs(u0(:))); % brings initial image satisfies -1<=u<=1
  [M, N] = size(u0);
 
% Add noise
  if noisy == 1
    sigma = 0.15;
    u0 = u0 + sigma*randn(size(u0));
%     u0(u0<0) = 0;
%     u0(u0>1) = 1;
  end
  
% Set edge indicator
  if edge == 0
    g = ones(size(u0));
  else
    g = imgaussfilt(u0) ;
    g = 1./(1 + imgradient(g).^2);
  end

% Visualize the (initial) image u0 in Matlab
  figure(fignum); clf; subplot(3,3,1)
  imagesc(u0); axis('image', 'off')
  title('\bf Original (noisy) image', 'fontsize', 20);

  
%% Initialize level set function
  x = linspace(1, N, N); 
  y = linspace(1, M, M);
  [X, Y] = meshgrid(x, y);
  
  switch u_type
    case 'circle'
      r = min(M,N)/4;
      u = -sqrt( (X-(N+1)/2).^2 + (Y-(M+1)/2).^2 ) + r;
      
    case {'sqr1', 'sqr3', 'sqr4', 'blur1', 'blur2', 'bar2' }
      u = -sqrt( (X-(N+1)/2).^2 + (Y-(M+1)/2).^2 ) + r;
      
    case 'sqr2'
      u = -sqrt( (X-75).^2 + (Y-75).^2 ) + 40;
      u = max(u, -sqrt( (X-160).^2 + (Y-160).^2 ) + 20);
      
    case 'sidebar'
      u = -sqrt( (X-(0)/2).^2 + (Y-(M+1)/2).^2 ) + r;
      
    case {'same', 'default'}
    % Set starting point to be the initial given image   
      u = u0;
  end
  % Set 0 <= u <= 1
  umax = max(u(:));
  umin = min(u(:));
  u = (u - umin)/(umax - umin);

  
%% %%%%%%%%%%    Begin iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter=1:iter_max
% Compute region average C1, C2:
  if nnz(u>thres) == 0
    C1 = ep2;
  else
    C1 = mean(u0(u>thres),'all');
  end
  if nnz(u<=thres) == 0
    C2 = ep2;
  else
    C2 = mean(u0(u<=thres),'all');
  end
  
  % Update pointwise (Gauss-Seidel type semi-implicit scheme)  
  for i = 2:M-1
    for j = 2:N-1
      ux = ( u(i+1,j) - u(i,j) )/h;
      uy = ( u(i,j+1) - u(i,j-1) )/(2*h);
      Gradu = sqrt( ep2 + ux*ux + uy*uy );
      co1 = g(i,j) ./ Gradu;

      ux = ( u(i,j) - u(i-1,j) )/h;
      uy = ( u(i-1,j+1) - u(i-1,j-1) )/(2*h);
      Gradu = sqrt( ep2 + ux*ux + uy*uy );
      co2 = g(i-1,j) ./ Gradu;

      ux = ( u(i+1,j) - u(i-1,j) )/(2*h);
      uy = ( u(i,j+1) - u(i,j) )/h;
      Gradu = sqrt( ep2 + ux*ux + uy*uy );
      co3 = g(i,j) ./ Gradu;

      ux = ( u(i+1,j-1) - u(i-1,j-1) )/(2*h);
      uy = ( u(i,j) - u(i,j-1) )/h;
      Gradu = sqrt( ep2 + ux*ux + uy*uy );
      co4 = g(i,j-1) ./ Gradu;
      
      co = 1.0 + (dt/h^2)*( co1+co2+co3+co4 );
      
    	div = co1*u(i+1,j) + co2*u(i-1,j) + co3*u(i,j+1) + co4*u(i,j-1);
      s = ( C1-u0(i,j) )^2 - ( C2-u0(i,j) )^2;
      
      u(i,j) = (1./co) * ( u(i,j) + dt*( (1/h^2)*div - lambda*s ...
        - alpha*vp( u(i,j) ) ) );
    end
  end
  % End pointwise updates
  
% Update boundaries
  u = BCs(u, M, N);
 
% Stopping criteria; min 5 iterations
  if abs(C1-oldC1)/(C1+ep2)<tol && abs(C2-oldC2)/(C2+ep2)<tol && iter>5
    break;
  else
    oldC1 = C1;
    oldC2 = C2;
  end

% Mid-cycle updates
%   fprintf('Iter = %3d, C1 = %4.4g, C2 = %4.4g\n', iter, C1, C2);
  if mod(iter, 99) == 0    % change to small# for more updates
    plotseg(u0, u, fignum, lambda, C1, C2, iter, u_type);
  end
  
end 
% %%%%%%%%%%     End iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Plot final results
  fprintf('Iter = %3d, C1 = %4.4g, C2 = %4.4g\n', iter, C1, C2);
  plotseg(u0, u, fignum, lambda, C1, C2, iter, u_type);
  fprintf('\n');
end
% End of main function

function[] = plotseg(u0, u, fignum, lambda, C1, C2, Iter, u_type)
%% Visualize intermediate and final results 
%
  v = u;
  v(u>0.5) = 1;
  v(u<=0.5) = 0;
  figure(fignum); subplot(3,3,[2 3 5 6 8 9]);
  imagesc( u0 );  
  axis('image', 'off'); colormap(gray); hold on
  contour( v, [0.5 0.5], 'r', 'linewidth', 2.0 );
  hold off
  
  title({['\bf GAC -- ', u_type], ... 
    ['$\lambda$ = ', num2str(lambda), ', C1 = ', num2str(C1, '%3.4g'), ...
    ', C2 = ', num2str(C2, '%3.4g'), ' , Iter = ', num2str(Iter)]}, ...
    'fontsize', 20)
  
  subplot(3,3,[4 7])
  imagesc( u > 0.5 ); 
  axis('image', 'off'); colormap(gray);
  title('\bf u after thresholding' ,'fontsize', 20)
  
  h = gca; 
  h.FontSize = 18;
  h.TickLabelInterpreter = 'latex';
  
end

function[phi] = BCs(phi, M, N)
%% Sets BCs 
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

function[vprime] = vp(u)
%% Derivative of a regularize version of the penalty function, v. 
% See Figure 5 in [1].
  vep = 1e-4;
  vep2 = vep^2;
  
  if u <= -vep
    v = -2;
  elseif u > -vep && u < 0   % v = 1/(veps)*( u-veps );
    v = (2/vep2)*u^2*(u - 2*vep);    
  elseif u >= 0 && u <= 1
    v = 0;
  elseif u > 1 && u < 1+vep  % v = 1/(veps)*( u-1+eps );
    v = (-2/vep2)*(u-1)^2*(u-1-2*vep);
  else
    v = 2;
  end
  
  vprime = v;
end





  