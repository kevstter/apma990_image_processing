function[u,u0,E] = gcs_ps_sb(varargin)
%% gcs_ps_sb(im, lambda, edge, noisy, iter_max, fignum)
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
% Image segmentation by the model introduced in [3] (convexification of 
% ACWE model [1,2] for piecewise smooth 2-phase seg). The energy to be 
% minimized is
%   E = integrate g*agrad(u) + integrate mu*r(x)*u,
% where agrad(u) = abs(grad(u)) and g is an edge indicator function.
% This code implements the split Bregman [4] to minimize E.
%
% Eg:
% >> ims = {'sqr2','sqr4','bar','sidebar','blur','blur2','target','cam'};
% >> for k=1:length(ims), gcs_ps_sb(ims{k},1,0,1,50,120); pause(0.5); end
%
% >> [u,~,E] = gcs_ps_sb;
% >> figure(1); subplot(2,1,1); hist(u); subplot(2,1,2); plot(E);
%
% References:
% [1] VeseChan, A Multiphase LS framework for image segmentation...(2002)
% [2] ChanVese, Active contours without edges (2001)
% [3] BressonEsedogluVanderheynstThiranOsher, Fast global minim...(2007)
% [4] GoldsteinBressonOsher, Geometric application of the split B...(2010)
%
% Created: 24Apr2020
% Last modified: 24Apr2020
%

%% Read inputs: (im, mu, edge, noisy, iter_max, fignum)
  if nargin < 1
    fprintf('Default test example\n');
  end
  numvarargs = length(varargin);
  if numvarargs > 6
    error('Too many inputs...');
  end
  % mu, edge, noisy, iter_max, fignum, u_type
  optargs = {'bar', 1, 0, 1, 50, 120};
  optargs(1:numvarargs) = varargin;
  [im, mu, edge, noisy, iter_max, fignum] = optargs{:};

%% Set parameters  
% Model parameters
  lambda = 0.50;
  thresh = 0.50;
  eta = 1;
  h = 1;
 
% Misc
  tol = 9e-4; % stopping tol  
  
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
  
  S1 = u0;
  S2 = u0;
  dels1 = imgradient( S1, 'central' );
  dels2 = imgradient( S2, 'central' );
  r = (S1 - u0).^2 - (S2 - u0).^2 + eta*dels1.^2 - eta*dels2.^2;
  E = zeros(1, iter_max);
  
  [dy, dx] = imgradientxy( u, 'intermediate' );
  bx = zeros(size(u));
  by = zeros(size(u));

% Show image
  figure(fignum); clf; subplot(2,3,1)
  imagesc(u0); axis('image', 'off')
  title('\bf Original image', 'fontsize', 20);  
  
%% %%%%%%%%%%    Begin iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter=1:iter_max

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
  
  % Compute region average S1, S2: % How often to update these??
  if mod( iter, 10 ) == 0 
    [S1, S2] = gets1s2(u, u0, thresh, S1, S2, eta, h);
    dels1 = imgradient( S1, 'central' );
    dels2 = imgradient( S2, 'central' );
    r = (S1 - u0).^2 - (S2 - u0).^2 + eta*dels1.^2 - eta*dels2.^2;
  end
  
% Stopping criteria; min 5 iterations
  E(iter) = discrete_E(u, g, mu, r);
  if iter>4 && abs( E(iter)-E(iter-1) )/abs(E(iter)) < tol
    E = E(1:iter);
    break;
  end
  
% Mid-cycle updates
%   fprintf('Iter = %3d, C1 = %4.4g, C2 = %4.4g\n', iter, C1, C2);
  if mod(iter, 1000) == 0    % change to small# for more updates
    plotseg(u0, u, fignum, mu, S1, S2, iter, thresh);
  end
end 
% %  End iterations   % %


%% Plot final results
  fprintf('\n');
  plotseg(u0, u, fignum, mu, S1, S2, iter, thresh);
end
% % End of main function % %

function[] = plotseg(u0, u, fignum, mu, S1, S2, Iter, thresh)
%% Visualize intermediate and final results 
  v = zeros(size(u));
  v(u>thresh) = 1;
  figure(fignum); subplot(2,3,2);
  imagesc( u ); axis('image','off'); colormap(gray); hold on 
  contour( v, [thresh thresh], 'r', 'linewidth', 2.0 );
  hold off
  title({'\bf $u$ (no thresholding) + contour', ... 
    ['$\mu$ = ', num2str(mu), ' , Iter = ', num2str(Iter)]}, ...
    'fontsize', 20)
  
  subplot(2,3,3)
  imagesc( S2.*v + S1.*(1-v) ); axis('image', 'off'); colormap(gray);
  title({'\bf Split Bregman -- piecewise smooth approx:', ...
    'S2*($u>$thresh)+S1*($u<=$thresh)'} , ...
    'fontsize', 20)
  
  subplot(2,3,4)
  imagesc( S1 ); axis('image', 'off'); colormap(gray);
  title('\bf S1' ,'fontsize', 20)
  
  subplot(2,3,5)
  imagesc( S2 ); axis('image', 'off'); colormap(gray);
  title('\bf S2' ,'fontsize', 20)
  
  subplot(2,3,6)
  imagesc( abs( S1 - S2 ) ); axis('image', 'off'); colormap(gray);
  title('\bf abs(S1 - S2)', 'fontsize', 20);
  
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

  