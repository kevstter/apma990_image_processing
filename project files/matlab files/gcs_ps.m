function[u,u0,E] = gcs_ps(varargin)
%% gcs_ps(im, lambda, edge, noisy, iter_max, fignum)
% Inputs:
%   im: image; or string to select from default test cases. see init_im.m
%   lambda: reg parameter
%   edge: 0 or 1; 
%   noisy: 1 or 0
%   iter_max: max # of iterations
%   fignum: figure# for plotting
% Output:
%   u: gradient descent solution; thresholding not yet applied
%   u0: initial (noisy) image
%   E: discrete energy
%
% Image segmentation by the model introduced in [4] (convexification of 
% ACWE model [3,2] for piecewise smooth 2-phase seg). 
% Gradient descent with semi-implicit discretization similar to [2]. 
% The energy to be minimized is 
%     integrate g*agrad(u) + integrate lambda*r(x) + alpha*v(u),
% and its Euler-Lagrange equation
%     u_t = div( g*grad(u)/agrad(u) ) - lambda*r(x) - alpha*v'(u)
% where agrad(u) = abs(grad(u)), and g is an edge indicator function.
%
% Eg:
% >> ims = {'sqr2','sqr4','bar','sidebar','blur','blur2','target','cam'};
% >> for k=1:length(ims), gcs_ps(ims{k},1,0,1,60,110); pause(0.5); end
%
% >> [u,~,E] = gcs_ps;
% >> figure(1); subplot(2,1,1); hist(u); subplot(2,1,2); plot(E);
%
% References:
% [1] ChanEsedogluNikolova, Algos for finding global minimizers...(2006)
% [2] ChanVese, Active contours without edges (2001)
% [3] VeseChan, A Multiphase LS framework for image segmentation...(2002)
% [3] BressonEsedogluVanderheynstThiranOsher, Fast global minim...(2007)
% [4] PinarZenios, On smoothing exact penalty functions for...(1994)
%
% Created: 21Apr2020
% Last modified: 24Apr2020
%

%% Read inputs: (im, lambda, edge, noisy, iter_max, fignum)
  if nargin < 1
    fprintf('Default test example\n')
  end
  numvarargs = length(varargin);
  if numvarargs > 6
    error('Too many inputs...');
  end
  % lambda, edge, noisy, iter_max, fignum, u_type
  optargs = {'bar', 1, 0, 1, 50, 110};
  optargs(1:numvarargs) = varargin;
  [im, lambda, edge, noisy, iter_max, fignum] = optargs{:};

%% Set parameters
% Space & time discretization 
  h = 1.0;
  dt = 5e-4;
  
% Model parameters
  eta = 1;
  alpha = lambda;
  thresh = 0.50;
  
% Regularize TV at the origin 
  ep = 1e-6;
  ep2 = ep*ep;
  
% Misc
  tol = 1e-3; % stopping tol  
  
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
  
%% Initialize level set function
  u = u0;
%   u = randn(size(u0));
  umax = max(u(:));
  umin = min(u(:));
  u = (u - umin)/(umax - umin);
  
  S1 = u0; 
  S2 = u0;
  dels1 = imgradient( S1, 'central' );
  dels2 = imgradient( S2, 'central' );
  r = (S1 - u0).^2 - (S2 - u0).^2 + eta*dels1.^2 - eta*dels2.^2;
  E = zeros(1,iter_max);

% Show image
  figure(fignum); clf; subplot(2,3,1)
  imagesc(u0); axis('image', 'off')
  title('\bf Original image', 'fontsize', 20);  
  
%% %%%%%%%%%%    Begin iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter=1:iter_max
  
  vsp = vsprime( u );
  
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
      
      u(i,j) = (1./co) * ( u(i,j) + dt*( (1/h^2)*div - lambda*r(i,j) ...
        - alpha*vsp(i,j) ) );
      
    end
  end
  % End pointwise updates
  
% Update boundaries
  u = BCs(u, M, N);
  
  % Compute region average S1, S2: % How often to update these??
  if mod( iter, 10 ) == 0 
    [S1, S2] = gets1s2(u, u0, thresh, S1, S2, eta, h);
    dels1 = imgradient( S1, 'central' );
    dels2 = imgradient( S2, 'central' );
    r = (S1 - u0).^2 - (S2 - u0).^2 + eta*dels1.^2 - eta*dels2.^2;
  end
  
% Stopping criteria; min 5 iterations
  E(iter) = discrete_E(u, g, lambda, r, alpha);
  if iter>4 && abs( E(iter)-E(iter-1) )/abs(E(iter)) < tol
    E = E(1:iter);
    break;
  end

% Mid-cycle updates
  if mod(iter, 1000) == 0    % change to small# for more updates
    plotseg(u0, u, fignum, lambda, S1, S2, iter, thresh);
  end
  
end 
% %  End iterations   % %


%% Plot final results
%   fprintf('Iter = %3d, C1 = %4.4g, C2 = %4.4g\n', iter, S1, S2);
  fprintf('\n');
  plotseg(u0, u, fignum, lambda, S1, S2, iter, thresh);
end
% % End of main function % %

function[] = plotseg(u0, u, fignum, lambda, S1, S2, Iter, thresh)
%% Visualize intermediate and final results
%
  v = zeros(size(u));
  v(u>thresh) = 1;
  figure(fignum); subplot(2,3,2);
  imagesc( u );  axis('image', 'off'); colormap(gray); hold on
  contour( v, [thresh thresh], 'r', 'linewidth', 2.0 );
  hold off
  title({'\bf $u$ (no thresholding) + contour', ... 
    ['$\lambda$ = ', num2str(lambda), ' , Iter = ', num2str(Iter)]}, ...
    'fontsize', 20)
  
  subplot(2,3,3)
  imagesc( S2.*v + S1.*(1-v) ); axis('image', 'off'); colormap(gray);
  title({'\bf Piecewise smooth approx:', ...
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

function[E] = discrete_E(u, g, lambda, s, alpha)
%% Compute discrete energy
%
  dmag = imgradient(u, 'central');
  vs = vsmooth(u);
  E = sum( g.*dmag, 'all' ) + lambda*sum( s.*u, 'all' ) + ...
    alpha*sum( vs, 'all' );
end

function[vs] = vsmooth(u)
%% smoothed penalty function
% See Figure 5 in [1]. See [4]
%
  vep = 1e-9;   % should be same values as in vsprime(u)
  vs = zeros(size(u));
  
  cond1 = (u<=-vep/2);
  cond2 = (u>-vep/2) & (u<0);
  cond3 = (u>=0) & (u<=1);
  cond4 = (u>1) & (u<1+vep/2);
  cond5 = (u>=1+vep/2);
  
  vs(cond1) = -2*u(cond1)-vep/2;
  vs(cond2) = (2/vep)*u(cond2).^2;
  vs(cond3) = 0;
  vs(cond4) = (2/vep)*(u(cond4)-1).^2;
  vs(cond5) = 2*(u(cond5)-1)-vep/2;

end

function[vsp] = vsprime(u)
%% Derivative of smoothed penalty function, v. 
% See Figure 5 in [1]. See [4].
%
  vep = 1e-9;   % should be same value as in vsmooth(u)
  vsp = zeros(size(u));
  
  cond1 = (u<=-vep/2);
  cond2 = (u>-vep/2) & (u<0);
  cond3 = (u>=0) & (u<=1);
  cond4 = (u>1) & (u<1+vep/2);
  cond5 = (u>=1+vep/2);
  
  vsp(cond1) = -2;
  vsp(cond2) = 4*u(cond2)/vep;
  vsp(cond3) = 0;
  vsp(cond4) = 4*( u(cond4)-1 )/vep;
  vsp(cond5) = 2;
  
end
