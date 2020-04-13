function[u,u0,E] = gcs_sb(varargin)
%% gcs_sb(im, lambda, edge, noisy, iter_max, fignum)
% Inputs:
%   im: image; or string to select from default test cases. see init_im.m
%   lambda: model parameter
%   edge: 0 or 1; 
%   noisy: 1 or 0
%   iter_max: max # of iterations
%   fignum: figure# for plotting
% Output:
%   u: gradient descent solution; thresholding not yet applied
%   u0: initial (noisy) image
%   E: energy 
%
% Image segmentation by the model introduced in [1] (convexification of 
% ACWE model [2] for 2-phase seg) and advanced in [3]. The energy to be 
% minimized is
%   E = integrate g*agrad(u) + integrate mu*r(x)*u,
% where agrad(u) = abs(grad(u)) and g is an edge indicator function.
% This code implements the split Bregman [4] to minimize E.
%
% Eg:
% >> ims = {'sqr2','sqr3','sqr4','bar','sidebar','blur','blur2','default'};
% >> for k=1:length(ims), gcs(ims{k},10,1,1,50,90); pause(0.5); end
%
% References:
% [1] ChanEsedogluNikolova, Algos for finding global minimizers...(2006)
% [2] ChanVese, Active contours without edges (2001)
% [3] BressonEsedogluVanderheynstThiranOsher, Fast global minim...(2007)
% [4] GoldsteinBressonOsher, Geometric application of the split B...(2010)
%
% Created: 30Mar2020
% Last modified: 13Apr2020
%

%% Read inputs: (im, mu, edge, noisy, iter_max, fignum)
  if nargin < 1
    fprintf('Default test example\n');
  end
  numvarargs = length(varargin);
  if numvarargs > 5
    error('Too many inputs...');
  end
  % mu, edge, noisy, iter_max, fignum, u_type
  optargs = {'grid', 1e-0, 0, 0, 500, 100};
  optargs(1:numvarargs) = varargin;
  [im, mu, edge, noisy, iter_max, fignum] = optargs{:};

%% Set parameters  
% Model parameters
  lambda = 0.50;
  thresh = 0.50;
 
% Misc
  tol = 9e-5; % stopping tol  
  
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
    sigma = 0.10;
    u0 = u0 + sigma*randn(size(u0));
  end
  
% Set edge indicator
  if edge == 0
    g = ones(size(u0));
  else
    g = imgaussfilt(u0) ;
    g = 1./(1 + imgradient(g).^2);
  end
  
%% Initialize u, d=grad(u), 'bregman param b
  u = u0;
  
  [C1,C2] = getc1c2(u, u0, thresh);
  E = zeros(1, iter_max);
  
  [dy, dx] = imgradientxy( u, 'intermediate' );
  bx = zeros(size(u));
  by = zeros(size(u));

% Show image
  figure(fignum); clf; subplot(3,3,1)
  imagesc(u0); axis('image', 'off')
  title('\bf Original (noisy) image', 'fontsize', 20);  
  
%% %%%%%%%%%%    Begin iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter=1:iter_max
  
  r = (C1 - u0).^2 - (C2 - u0).^2;

  % Gauss-Seidel pointwise updates
  for i=2:M-1
    for j=2:N-1
      alpha = dx(i-1,j) - dx(i,j) - bx(i-1,j) + bx(i,j) ...
        + dy(i,j-1) - dy(i,j) - by(i,j-1) + by(i,j);
      beta = 0.25*( u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1)  ...
        - (mu/lambda)*r(i,j) + alpha);
      u(i,j) = max( min(beta,1), 0 );
    end
  end
  
  u = BCs(u, M, N);
  
  % Update d
  [uy, ux] = imgradientxy( u, 'intermediate' );
  sx = bx + ux; 
  sy = by + uy; 
  s = sum( sx.^2 + sy.^2, 'all' ); s = sqrt(s);
  sz = max( s-g./lambda, 0 );
  dx = sz.*sx/s;
  dy = sz.*sy/s;
  dx = BCs(dx, M, N);
  dy = BCs(dy, M, N);
  
  % Update b
  bx = bx - dx + ux;
  by = by - dy + uy;
  bx = BCs(bx, M, N);
  by = BCs(by, M, N);
  
  % Compute region average
  [C1, C2] = getc1c2(u, u0, thresh);
  
% Stopping criteria; min 5 iterations
  E(iter) = discrete_E(u, g, mu, r);
  if iter>4 && abs( E(iter)-E(iter-1) )/abs(E(iter)) < tol
    E = E(1:iter);
    break;
  end
  
% Mid-cycle updates
%   fprintf('Iter = %3d, C1 = %4.4g, C2 = %4.4g\n', iter, C1, C2);
  if mod(iter, 100) == 0    % change to small# for more updates
    plotseg(u0, u, fignum, mu, C1, C2, iter);
  end
end 
% %  End iterations   % %


%% Plot final results
  fprintf('Iter = %3d, C1 = %4.4g, C2 = %4.4g\n', iter, C1, C2);
  plotseg(u0, u, fignum, mu, C1, C2, iter);
end
% % End of main function % %

function[] = plotseg(u0, u, fignum, mu, C1, C2, Iter)
%% Visualize intermediate and final results 
  v = u;
  v(u>0.5) = 1;
  v(u<=0.5) = 0;
  figure(fignum); subplot(3,3,[2 3 5 6 8 9]);
  imagesc( u0 );  axis('image', 'off'); colormap(gray); hold on
  contour( v, [0.5 0.5], 'r', 'linewidth', 2.0 );
  hold off
  
  title({'\bf GAC with split Bregman', ... 
    ['$\mu$ = ', num2str(mu), ', C1 = ', num2str(C1, '%3.4g'), ...
    ', C2 = ', num2str(C2, '%3.4g'), ' , Iter = ', num2str(Iter)]}, ...
    'fontsize', 20)
  
  subplot(3,3,[4 7])
  imagesc( v ); axis('image', 'off'); colormap(gray);
  title('\bf After thresholding u' ,'fontsize', 20)
  
  h = gca; 
  h.FontSize = 18;
  h.TickLabelInterpreter = 'latex';
  
end

function[E] = discrete_E(u, g, mu, r)
%% Compute discrete energy
% 
  dmag = imgradient(u, 'central');
  E = sum( g.*dmag, 'all' ) + mu*sum( r.*u, 'all');
end

  