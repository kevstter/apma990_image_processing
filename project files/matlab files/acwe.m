function[phi, u0, E] = acwe(varargin)
%% acwe(im, mu, noisy, iter_max, fignum, phi_type)
% Inputs:
%   im: image; or string to select from default test cases. see init_im.m
%   lambda: regularization parameter
%   noisy: 1 or 0
%   iter_max: max # of iterations
%   fignum: figure# for plotting
%   phi_type: level set function initialization
% Output:
%   phi: level set function
%   u0: initial image
%   E: energy
%
% Matlab code modified from 
%     http://www.math.ucla.edu/~lvese/285j.1.05s/TV_L2.m
%
% Image segmentation by the semi-implicit gradient descent algorithm as 
% described in 
%     Chan and Vese, Active contours without edges (2001).
%
% Eg:
% >> ims = {'sqr2','sqr4','bar','sidebar','blur','blur2','target','cam'};
% >> for k=1:length(ims), acwe(ims{k},20,0,50,80+k); pause(0.5); end
%
% References:
% [2] ChanVese, Active contours without edges (2001)
%
% Last modified: 19Apr2020
%

%% Read inputs: (im, lambda, noisy, iter_max, fignum, phi_type)  
  if nargin < 1
    fprintf('Default test example\n')
  end
  numvarargs = length(varargin);
  if numvarargs > 6
    error('Too many inputs...');
  end
  % im, lambda, noisy, iter_max, fignum, phi_type
  optargs = {'grid', 40, 0, 50, 80, 'bubbles'};
  optargs(1:numvarargs) = varargin;
  [im, lambda, noisy, iter_max, fignum, phi_type] = optargs{:};
  
%% Set parameters
% Space & time discretization 
  h = 1.0;
  dt = 3e-2;
  
% Model parameters
  mu = 1;
  alpha = mu/h^2;
  nu = 0;

% Regularize TV at the origin 
  eps = 1e-6;
  ep2 = eps*eps;
  
% Stopping crit
  tol = 9e-5;
  E = zeros(1,iter_max);
  
%% Load initial data
  if isa( im, 'char' ) == 1
    % Variety of simple synthetic images for testing
    [u0, r] = init_im( im );   
  else
    u0 = im2double(im);
    r = 1;
  end
  u0 = u0/max(abs(u0(:)));
  [M, N] = size(u0);
 
% Add noise
  if noisy == 1
    sigma = 0.10;
    u0 = u0 + sigma*randn(size(u0));
  end
  
%% Initialize level set function
  [phi, ~, ~] = init_ls( N, M, r, phi_type );
  phi = phi/max(abs(phi(:)));
%   oldphi = phi;
  
  [C1, C2] = getc1c2(phi, u0, 0);
  
%% Show image and initial contour
  init_plot(fignum, u0, phi);
  
%% %%%%%%%%%%    Begin iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter=1:iter_max
  
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
      ss = (u0(i,j)-C1)^2 - (u0(i,j)-C2)^2;
      
    	div = co1*phi(i+1,j) + co2*phi(i-1,j) + co3*phi(i,j+1) + co4*phi(i,j-1);
      
      phi(i,j) = (1./co) * ( phi(i,j) + dt*delh*( alpha*div - nu ...
        - lambda*( ss ) ) );
      
    end
  end
  % End pointwise updates
  
% Update boundaries
  phi = BCs(phi, M, N);
 
% Compute region average C1, C2:
  [C1, C2] = getc1c2(phi, u0, 0);

% Stopping criteria: to do--switch to discrete energy stopping
%   if iter>4 && norm( oldphi-phi, 'fro' )/numel(phi) < tol
  E(iter) = discrete_E(phi, u0, C1, C2, lambda, mu, h);
  if iter>24 && abs( E(iter)-E(iter-1) )/abs(E(iter)) < tol
    E = E(1:iter);
    break;
  end
%   oldphi = phi;

% Mid-cycle plot updates
  if mod(iter, 50) == 0    % change 500 to small# for more updates
    plotseg(u0, phi, fignum, lambda, C1, C2, iter);
  end
  
end 
% %%%%%%%%%%     End iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot final results
  fprintf('Iter = %3d, C1 = %4.4g, C2 = %4.4g\n', iter, C1, C2);
  plotseg(u0, phi, fignum, lambda, C1, C2, iter);
end
% % End of main function % %

function[E] = discrete_E(u, u0, C1, C2, lambda, mu, h)
%% Compute discrete energy
%
  del = h ./ ( pi*(h^2 + u.^2) );
  gmag = imgradient(u, 'central');
  E1 = mu*sum( del.*gmag, 'all' );
  E2 = lambda*sum( (u0(u>0) - C1).^2, 'all' );
  E3 = lambda*sum( (u0(u<0) - C2).^2, 'all' );
  E = E1 + E2 + E3;
  
end

function[] = plotseg(u0, phi, fignum, lambda, C1, C2, Iter)
%% Visualize intermediate and final results 
%
  figure(fignum); subplot(3,3,[2 3 5 6 8 9]);
  imagesc(u0);  axis('image', 'off'); colormap(gray); 
  hold on 
  contour(phi, [0,0], 'linewidth', 2.0, 'linecolor', 'r');
  hold off;
  
  title({'\bf Active contours without edges ', ... 
    ['$\mu$ = ', num2str(lambda), ', C1 = ', num2str(C1, '%3.4g'), ...
    ', C2 = ', num2str(C2, '%3.4g'), ' , Iter = ', num2str(Iter)]}, ...
    'fontsize', 20)

  h = gca; 
  h.FontSize = 18;
  h.TickLabelInterpreter = 'latex';
  
end

function[] = init_plot(fignum, u0, phi)
% Show image+initial contour
  figure(fignum); clf; 
  subplot(3,3,1)
  imagesc(u0); axis('image', 'off')
  title('\bf Original (noisy) image)', 'fontsize', 20);
  subplot(3,3,[4 7])
  imagesc(u0); axis('image', 'off'); hold on 
  contour(phi, [0 0], 'r', 'linewidth', 2.0); hold off
  title('\bf Image + initial contour', 'fontsize', 20);
  
end