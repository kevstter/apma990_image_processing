function[phi, u0] = acwe(im, varargin)
%% acwe(im, mu, noisy, iter_max, fignum, phi_type)
% Inputs:
%   im: select synthetic image (see options below) for segmentation
%   lambda: regularization parameter
%   noisy: 1 or 0
%   iter_max: max # of iterations
%   fignum: figure# for plotting
%   phi_type: level set function initialization
% Output:
%   phi: level set function
%   im: initial image
%
% Matlab code modified from 
%     http://www.math.ucla.edu/~lvese/285j.1.05s/TV_L2.m
%
% Image segmentation by the semi-implicit gradient descent algorithm as 
% described in 
%     Chan and Vese, Active contours without edges (2001).
%
% Eg:
% >> ims = {'sqr2','sqr3','sqr4','bar','sidebar','blur','blur2','default'};
% >> for k=1:length(ims), acwe(ims{k},20,0,50,80+k); pause(0.5); end
%
% Last modified: 29Mar2020
%

%% Read inputs: (im, lambda, noisy, iter_max, fignum, phi_type)  
  if nargin < 1
    error('Missing all inputs');
  end
  numvarargs = length(varargin);
  if numvarargs > 6
    error('Too many inputs...');
  end
  % lambda, noisy, iter_max, fignum, phi_type
  optargs = {40, 0, 50, 80, 'bubbles'};
  optargs(1:numvarargs) = varargin;
  [lambda, noisy, iter_max, fignum, phi_type] = optargs{:};

%% Set parameters
% Space & time discretization 
  h = 1.0;
  dt = 9e-3;
  
% Model parameters
%   lambda = 1; 
  mu = 1;
  alpha = mu/h^2;
  nu = 0;

% Regularize TV at the origin 
  eps = 1e-6;
  ep2 = eps*eps;
  
%% Load initial data
  if isa( im, 'char' ) == 1
    [u0, r] = init_im( im );   
  else
    u0 = im2double(im);
    r = 1;
  end
  [M, N] = size(u0);
%   u0 = u0/max(abs(u0(:)))*255;
 
% Add noise
  if noisy == 1
    sigma = 15;
    u0 = u0 + sigma*randn(size(u0));
  end
  u0 = u0/max(abs(u0(:)));

%% Initialize level set function
  [phi, ~, ~] = init_ls( N, M, r, phi_type );
  phi = phi/max(abs(phi(:)));
  
  [C1, C2] = getc1c2(phi, u0, 0);
  
%% Show image and initial contour
  init_plot(fignum, u0, phi);
  
%% Others things to initialize
  tol = 1e-2;                           % stopping tol
  
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
  fprintf('Iter = %3d, C1 = %8.9g, C2 = %3.8g\n', iter, C1, C2);

% Mid-cycle plot updates
  if mod(iter, iter_max/12) == 0    % change 500 to small# for more updates
    plotseg(u0, phi, fignum, lambda, C1, C2, iter);
  end
  
end 
% %%%%%%%%%%     End iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot final results
  plotseg(u0, phi, fignum, lambda, C1, C2, iter);
end
% % End of main function % %

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