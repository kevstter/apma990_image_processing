function[phi] = acwe(im, mu, iter_max, fignum, phi_type, noisy)
%% Matlab code modified from 
%     http://www.math.ucla.edu/~lvese/285j.1.05s/TV_L2.m
%
% Image segmentation by the semi-implicit gradient descent algorithm as 
% described in 
%     Chan and Vese, Active contours without edges (2001).
% 
% Inputs:
%   img: image for segmentation
%   mu: regularization parameter
%   iter_max: max # of iterations
%   phi_type: 2 options for initialization 'circle' or 'bubbles'
%   noisy: 1 or 0
% 
% Output:
%   phi: level set function
%
% Eg.
% >> acwe('sqr4', 100, 100, 31, 'bubbles', 1);
%
% Last modified: 27Mar2020
%
 

%% Read inputs
  if nargin < 1
    error('Missing inputs');
  elseif nargin < 6
    noisy = 0;
    if nargin < 5
      phi_type = im;
      if nargin < 4
        fignum = 96;
        if nargin < 3
          iter_max = 100;
          if nargin < 2
            mu = 100;
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
    
    otherwise
      fprintf('Default u0');
  end
%   
  [M, N] = size(u0);
 
% Add noise
  if noisy == 1
    u0 = u0 + 10*randn(size(u0));
  end

% Visualize the (initial) image u0 in Matlab
  figure(fignum); clf; subplot(3,3,1)
  imagesc(u0); axis('image', 'off')
  title('\bf Brain scan - original', 'fontsize', 20);

  
%% Set parameters
% Space & time discretization 
  h = 1.0;
  dt = 0.1;
  
% Model parameters
  lambda1 = 1; 
  lambda2 = 1;
  alpha = mu/h^2;
  nu = 0;

% Regularize TV at the origin 
  eps = 1e-6;
  ep2 = eps*eps;
  
  
%% Initialize level set function
  x = linspace(1, N, N); 
  y = linspace(1, M, M);
  [X, Y] = meshgrid(x, y);
  
  % Option 1: one circle 
  switch phi_type
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
      
    case {'sqr1', 'sqr3', 'sqr4', 'blur1', 'blur2', 'bar2' }
%       r = 75; %min(M,N)/4;
      phi = -sqrt( (X-(N+1)/2).^2 + (Y-(M+1)/2).^2 ) + r;
      
    case 'sqr2'
      phi = -sqrt( (X-75).^2 + (Y-75).^2 ) + 40;
      phi = max(phi, -sqrt( (X-160).^2 + (Y-160).^2 ) + 20);
      
    case 'sidebar'
      phi = -sqrt( (X-(0)/2).^2 + (Y-(M+1)/2).^2 ) + r;
  end
  phi = phi/max(abs(phi(:)));
  
  
%% Others things to init
  integ_u0 = trapz(y, trapz(x, u0, 2)); % integrate u0 dx
  integ_1 = prod(size(u0) - 1);         % integrate 1 dx
  Cold1 = -1; C1 = -1;                  % "initial" region avg: c1 and c2
  Cold2 = -1; C2 = -1;
  tol = 1e-2;                           % stopping tol
    
  
%% %%%%%%%%%%    Begin iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for Iter=1:iter_max
% Compute region average C1, C2:
  H = 0.5 + (1/pi)*atan( phi./h ) ;         % Heaviside function
  integ_H = trapz(y, trapz(x, H, 2));       % integrate H dx
  integ_u0H = trapz(y, trapz(x, u0.*H, 2)); % integrate u0*H dx');
  C1 = integ_u0H / integ_H;                 % region avg, c1 and c2
  C2 = ( integ_u0 - integ_u0H ) / ( integ_1 - integ_H );
%   thres = 0.5;
%   C1 = mean(u0(H>thres),'all');
%   C2 = mean(u0(H<=thres),'all');
  
  % Update pointwise (Gauss-Seidel type semi-implicit scheme)  
  for i = 2:M-1
    for j = 2:N-1
      phix = ( phi(i+1,j) - phi(i,j) )/h;
      phiy = ( phi(i,j+1) - phi(i,j-1) )/(2*h);
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co1 = 1 ./ Gradu;

      phix = ( phi(i,j) - phi(i-1,j) )/h;
      phiy = ( phi(i-1,j+1) - phi(i-1,j-1) )/(2*h);
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co2 = 1 ./ Gradu;

      phix = ( phi(i+1,j) - phi(i-1,j) )/(2*h);
      phiy = ( phi(i,j+1) - phi(i,j) )/h;
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co3 = 1 ./ Gradu;

      phix = ( phi(i+1,j-1) - phi(i-1,j-1) )/(2*h);
      phiy = ( phi(i,j) - phi(i,j-1) )/h;
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co4 = 1 ./ Gradu;
      
      delh = h / ( pi*(h^2 + phi(i,j)^2) );
      co = 1.0 + dt*delh*alpha*( co1+co2+co3+co4 );
      
    	div = co1*phi(i+1,j) + co2*phi(i-1,j) + co3*phi(i,j+1) + co4*phi(i,j-1);

      phi(i,j) = (1./co) * ( phi(i,j) + dt*delh*( alpha*div - nu ...
        - lambda1*( u0(i,j)-C1 )^2 + lambda2*( u0(i,j)-C2 )^2 ) );
    end
  end
% End pointwise updates

  
  fprintf('Iter = %3d, C1 = %8.9g, C2 = %3.8g\n', Iter, C1, C2);
  
% Update boundaries
  phi = BCs(phi, M, N);
 
% Stopping criteria
  if abs(C1 - Cold1)/(C1+eps) < tol && abs(C2 - Cold2)/(C2+ep2) < tol && Iter>5
    break;
  else
    Cold1 = C1;
    Cold2 = C2;
  end

% Mid-cycle plot updates
  if mod(Iter, 8) == 0    % change 500 to small# for more updates
    plotseg(u0, phi, fignum, mu, C1, C2, Iter, phi_type);
  end
  
end 
% %%%%%%%%%%     End iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Plot final results
  plotseg(u0, phi, fignum, mu, C1, C2, Iter, phi_type);
  fprintf('\n');
end
% End of main function

function[] = plotseg(u0, phi, fignum, mu, C1, C2, Iter, phi_type)
%% Visualize final results 
%
  figure(fignum); subplot(3,3,[2 3 5 6 8 9]);
  imagesc(u0);  axis('image', 'off'); colormap(gray); 
  hold on 
  contour(phi, [0,0], 'linewidth', 2.5, 'linecolor', 'r');
  hold off;
  
  title({['\bf Active contours without edges -- ', phi_type], ... 
    ['$\mu$ = ', num2str(mu), ', C1 = ', num2str(C1, '%3.4g'), ...
    ', C2 = ', num2str(C2, '%3.4g'), ' , Iter = ', num2str(Iter)]}, ...
    'fontsize', 20)
  
  subplot(3,3,[4 7])
  contourf(flipud(phi), [0 0], 'linewidth', 2.5);
  axis('image', 'off'); colormap(gray);
  title('\bf Level set function' ,'fontsize', 20)
  
  h = gca; 
  h.FontSize = 18;
  h.TickLabelInterpreter = 'latex';
  
end

function[phi] = BCs(phi, M, N)
%% Sets BCs of level set function 
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

%{
 
  px = (phi(ipx,iy)-phi(imx,iy))./(2*h); px2 = px.^2;
  py = (phi(ix,ipy)-phi(ix,imy))/(2*h); py2 = py.^2;      
  pxx = (phi(ipx,iy)-2*phi(ix,iy)+phi(imx,iy))/h^2;
  pyy = (phi(ix,ipy)-2*phi(ix,iy)+phi(ix,imy))/h^2;
  pxy = (phi(ipx,ipy)+phi(imx,imy)-phi(imx,ipy)-phi(ipx,imy))/(4*h*h);
  den = px2+py2+ep2;
  den = den.*sqrt(den); 
  K=(pxx.*py2-2*px.*py.*pxy+pyy.*px2)./den;
  
  rhs = delh.*(mu*K - nu ...
    - lambda1*(u0(ix,iy)-C1).^2 + lambda2*(u0(ix,iy)-C2).^2);
  phi(ix,iy) = phi(ix,iy) + dt*rhs;
%}