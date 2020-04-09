function[u] = gcs(im, varargin)
%% gcs(im, lambda, edge, noisy, iter_max, fignum)
% Inputs:
%   im: select synthetic image (see options below) for segmentation
%   lambda: model parameter
%   edge: 0 or 1; 
%   noisy: 1 or 0
%   iter_max: max # of iterations
%   fignum: figure# for plotting
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
%     u_t = div( g*grad(u)/agrad(u) ) - lambda*s(x) - alpha*v'(u)
% where agrad(u) = abs(grad(u)) and g is an edge indicator function.
%
% Eg:
% >> ims = {'sqr2','sqr3','sqr4','bar','sidebar','blur','blur2','default'};
% >> for k=1:length(ims), gcs(ims{k},10,1,1,50,90); pause(0.5); end
%
% References:
% [1] ChanEsedogluNikolova, Algos for finding global minimizers...(2006)
% [2] ChanVese, Active contours without edges (2001)
% [3] BressonEsedogluVanderheynstThiranOsher, Fast global minim...(2007)
% [4] PinarZenios, On smoothing exact penalty functions for...(1994)
%
% Created: 26Mar2020
% Last modified: 08Apr2020
%

%% Read inputs: (im, lambda, edge, noisy, iter_max, fignum)
  if nargin < 1
    error('Missing all inputs');
  end
  numvarargs = length(varargin);
  if numvarargs > 5
    error('Too many inputs...');
  end
  % lambda, edge, noisy, iter_max, fignum, u_type
  optargs = {1, 0, 0, 50, 90};
  optargs(1:numvarargs) = varargin;
  [lambda, edge, noisy, iter_max, fignum] = optargs{:};

%% Set parameters
% Space & time discretization 
  h = 1.0;
  dt = 0.05;
  
% Model parameters
  alpha = lambda/2;
  thres = 0.50;
  oldC1 = -1; C1 = -1;  % "initial" region avg: c1 and c2
  oldC2 = -1; C2 = -1;
  
% Regularize TV at the origin 
  ep = 1e-6;
  ep2 = ep*ep;
  
% Misc
  tol = 1e-6; % stopping tol  
  
%% Load initial data
  if isa(im, 'char') == 1
    % Variety of simple synthetic images for testing
    u0 = init_im(im);
  else
    u0 = im2double(im);
  end
  u0 = u0 / max(abs(u0(:))); % brings initial image satisfies -1<=u<=1
  [M, N] = size(u0);
 
% Add noise
  if noisy == 1
    sigma = 0.15;
    u0 = u0 + sigma*randn(size(u0));
  end
  
% Set edge indicator
  if edge == 0
    g = ones(size(u0));
  else
    g = imgaussfilt(u0) ;
    g = 1./(1 + imgradient(g).^2);
  end
  
%% Initialize level set function
  u = u0;
  umax = max(u(:));
  umin = min(u(:));
  u = (u - umin)/(umax - umin);

% Show image
  figure(fignum); clf; subplot(3,3,1)
  imagesc(u0); axis('image', 'off')
  title('\bf Original (noisy) image', 'fontsize', 20);  
  
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
  if abs(C1-oldC1)/(C1+ep2)<tol && abs(C2-oldC2)/(C2+ep2)<tol && iter>4
    break;
  else
    oldC1 = C1;
    oldC2 = C2;
  end

% Mid-cycle updates
%   fprintf('Iter = %3d, C1 = %4.4g, C2 = %4.4g\n', iter, C1, C2);
  if mod(iter, 99) == 0    % change to small# for more updates
    plotseg(u0, u, fignum, lambda, C1, C2, iter);
  end
  
end 
% %  End iterations   % %


%% Plot final results
  fprintf('Iter = %3d, C1 = %4.4g, C2 = %4.4g\n', iter, C1, C2);
  plotseg(u0, u, fignum, lambda, C1, C2, iter);
end
% % End of main function % %

function[] = plotseg(u0, u, fignum, lambda, C1, C2, Iter)
%% Visualize intermediate and final results 
  v = u;
  v(u>0.5) = 1;
  v(u<=0.5) = 0;
  figure(fignum); subplot(3,3,[2 3 5 6 8 9]);
  imagesc( u0 );  axis('image', 'off'); colormap(gray); hold on
  contour( v, [0.5 0.5], 'r', 'linewidth', 2.0 );
  hold off
  
  title({'\bf GAC', ... 
    ['$\lambda$ = ', num2str(lambda), ', C1 = ', num2str(C1, '%3.4g'), ...
    ', C2 = ', num2str(C2, '%3.4g'), ' , Iter = ', num2str(Iter)]}, ...
    'fontsize', 20)
  
  subplot(3,3,[4 7])
  imagesc( v ); axis('image', 'off'); colormap(gray);
  title('\bf After thresholding u' ,'fontsize', 20)
  
  h = gca; 
  h.FontSize = 18;
  h.TickLabelInterpreter = 'latex';
  
end

  