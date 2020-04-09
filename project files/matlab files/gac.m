function[phi, u0] = gac(im, varargin)
%% gac(im, dt, c, noisy, iter_max, fignum, phi_type)
% Inputs:
%   im: select synthetic image (see options below) for segmentation
%   dt: timestep size
%   c: constant velocity in the normal dir; c>0 (expand), c<0 (contract)
%   noisy: 1 or 0
%   iter_max: max # of iterations
%   fignum: figure# for plotting
%   phi_type: level set function initialization
% Output:
%   phi: levet set function
%   u0: initial image
%
% Matlab code modified from 
%     http://www.math.ucla.edu/~lvese/285j.1.05s/TV_L2.m
%
% Image segmentation by geodesic active contours (GAC) model from [1] via 
% a semi-implicit discretization similar to [2]. The energy to be 
% minimized is 
%     integrate g( grad(I) ) ds
% and its Euler-Lagrange equation 
%     u_t = agrad(u) * div( g*grad(u)/agrad(u) ) + c*g*agrad(u)
%         = g*(c + kappa)*agrad(u) + <grad(u), grad(g)>,
% where agrad(u) = abs(grad(u)). Upwinding is applied to the 2nd term [3].
%
% Eg:
% >> ims = {'sqr1','sqr2','sqr3','sqr4','bar','sidebar','blur','blur2'};
% >> for k=1:length(ims), gac(ims{k},0.05,-9,0,100,70+k); pause(0.5); end
%
% References:
% [1] CassellKimmelSapiro, Geodesic active contours (1997)
% [2] ChanVese, Active contours without edges (2001)
% [3] MarquinaOsher, Explicit algorithms for a new time-dependent...(2000)
%
% LAST MODIFIED: 29Mar2020
%

%% Read inputs (im, dt, c, noisy, iter_max, fignum, phi_type)
  if nargin < 1
    error('Missing all inputs');
  end
  numvarargs = length(varargin);
  if numvarargs > 6
    error('Too many inputs...');
  end
  % dt, c, noisy, iter_max, fignum, phi_type
  optargs = {0.05, 0, 0, 100, 80, im};
  optargs(1:numvarargs) = varargin;
  [dt, c, noisy, iter_max, fignum, phi_type] = optargs{:};
    
%% Set parameters
% Space & time discretization 
  h = 1.0;

% Regularize TV at the origin 
  ep = 1e-6;
  ep2 = ep*ep;
  
% Stopping tol
  tol = 1e-3;
  n2 = -1;
  
%% Load initial data
% Read simple synthetic images for testing
  [u0, r] = init_im( im );
  [M, N] = size(u0);
  
  % Add noise % Does this work...? Terribly..
  if noisy == 1
    sigma = max(abs(u0(:)))/100;
    u0 = u0 + sigma*randn(size(u0));
    for i = 1:1
      u0 = imgaussfilt(u0,sigma);
    end
  end
  
%% Setup edge detector
  g = imgaussfilt( u0 );   % gaussian-smoothed image
  g = 1./(1 + imgradient( g ).^2);  % classic edge detector, p=2
  
%% Initialize level set function
  phi = init_ls( N, M, r , phi_type );
  
%% Show initial state
  init_plot(fignum, u0, phi);
  
%% %%%%%%%%%%    Begin iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter=1:iter_max  
% Update pointwise (Gauss-Seidel type scheme)  
  for i = 2:M-1
    for j = 2:N-1
%       pm = [0 -1 0 -1];
      pm = get_edge(iter); 
      pm1=pm(1); pm2=pm(2); pm3=pm(3); pm4=pm(4);
      phix = ( phi(i+1,j) - phi(i,j) )/h;
      phiy = ( phi(i,j+1) - phi(i,j-1) )/(2*h);
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co1 = g(i+pm1,j) ./ Gradu;

      phix = ( phi(i,j) - phi(i-1,j) )/h;
      phiy = ( phi(i-1,j+1) - phi(i-1,j-1) )/(2*h);
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co2 = g(i+pm2,j) ./ Gradu;

      phix = ( phi(i+1,j) - phi(i-1,j) )/(2*h);
      phiy = ( phi(i,j+1) - phi(i,j) )/h;
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co3 = g(i,j+pm3) ./ Gradu;

      phix = ( phi(i+1,j-1) - phi(i-1,j-1) )/(2*h);
      phiy = ( phi(i,j) - phi(i,j-1) )/h;
      Gradu = sqrt( ep2 + phix*phix + phiy*phiy );
      co4 = g(i,j+pm4) ./ Gradu;
      
      phix = ( phi(i+1,j) - phi(i-1,j) ) / (2*h);
      phiy = ( phi(i,j+1) - phi(i,j-1) ) / (2*h);
      agrad = sqrt( phix*phix + phiy*phiy + ep2 );
      co = 1.0 + dt*agrad*( co1+co2+co3+co4 );
      
    	div = co1*phi(i+1,j) + co2*phi(i-1,j) + co3*phi(i,j+1) ...
        + co4*phi(i,j-1);
      
      % Upwinding on second term
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
%   n1 = nnz(phi<0);
%   if iter > 10 && abs(n1 - n2) < 2
%     break;
%   else
%     n2 = n1;
%   end

% Mid-cycle plot updates
  if mod(iter, 200) == 1    % change 500 to small# for more updates
    plotseg(u0, phi, fignum, c, iter);
  end
  
end 
% %%%%%%%%%%     End iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot final results
  plotseg(u0, phi, fignum, c, iter);
%   fprintf('\n');
end
% End of main function

function[g] = get_edge(iter)
  iter = mod(iter, 4);
  switch iter
    case 1
      g = [0 -1 0 -1];
    case 2
      g = [0 -1 1 0];
    case 3
      g = [1 0 1 0];
    case 0
      g = [1 0 0 -1];
  end
end

function[] = plotseg(u0, phi, fignum, c, iter)
%% Visualize intermediate and final results 
%
  figure(fignum); subplot(3,3,[2 3 5 6 8 9]);
  imagesc(u0);  axis('image', 'off'); colormap(gray); 
  hold on 
  contour(phi, [0,0], 'linewidth', 2.0, 'linecolor', 'r');
  hold off;
  
  title({'\bf GAC', ['c = ', num2str(c), ' , Iter = ', num2str(iter)]}, ...
    'fontsize', 20)
  
  h = gca; 
  h.FontSize = 18;
  h.TickLabelInterpreter = 'latex';
  
end

function[] = init_plot(fignum, u0, phi)
% Plots image and initial contour
  figure(fignum); clf;
  subplot(3,3,1)
  imagesc(u0); axis('image', 'off');
  title('\bf Original image', 'fontsize', 20);
  subplot(3,3,[4 7])
  imagesc(u0); axis('image', 'off'); hold on
  contour(phi, [0 0], 'r', 'linewidth', 2.0); hold off
  title('\bf Image + initial contour', 'fontsize', 20);
end