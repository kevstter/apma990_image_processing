function[phi, im] = acwe(im, varargin)
%% acwe(im, mu, noisy, iter_max, fignum, phi_type)
% Inputs:
%   im: select synthetic image (see options below) for segmentation
%   mu: length regularization parameter
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

%% Read inputs: (im, mu, noisy, iter_max, fignum, phi_type)  
  if nargin < 1
    error('Missing all inputs');
  end
  numvarargs = length(varargin);
  if numvarargs > 6
    error('Too many inputs...');
  end
  % mu, noisy, iter_max, fignum, phi_type
  optargs = {40, 0, 50, 80, 'bubbles'};
  optargs(1:numvarargs) = varargin;
  [mu, noisy, iter_max, fignum, phi_type] = optargs{:};

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
  
%% Load initial data
  if isa( im, 'char' ) == 1
    [u0, r] = init_im( im );   
  else
    u0 = im2double(im);
    r = 1;
  end
  [M, N] = size(u0);
  u0 = u0/max(abs(u0(:)))*255;
 
% Add noise
  if noisy == 1
    sigma = 10;
    u0 = u0 + sigma*randn(size(u0));
  end

%% Initialize level set function
  [phi, x, y] = init_ls( N, M, r, phi_type );
  phi = phi/max(abs(phi(:)));
  
%% Show image and initial contour
  init_plot(fignum, u0, phi);
  
%% Others things to initialize
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
  
% Update boundaries
  phi = BCs(phi, M, N);
 
% Stopping criteria
%   fprintf('Iter = %3d, C1 = %8.9g, C2 = %3.8g\n', Iter, C1, C2);
  if abs(C1 - Cold1)/(C1+eps) < tol && abs(C2 - Cold2)/(C2+ep2) < tol && Iter>5
    break;
  else
    Cold1 = C1;
    Cold2 = C2;
  end

% Mid-cycle plot updates
  if mod(Iter, 12) == 0    % change 500 to small# for more updates
    plotseg(u0, phi, fignum, mu, C1, C2, Iter);
  end
  
end 
% %%%%%%%%%%     End iterations    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot final results
  plotseg(u0, phi, fignum, mu, C1, C2, Iter);
end
% % End of main function % %

function[] = plotseg(u0, phi, fignum, mu, C1, C2, Iter)
%% Visualize intermediate and final results 
%
  figure(fignum); subplot(3,3,[2 3 5 6 8 9]);
  imagesc(u0);  axis('image', 'off'); colormap(gray); 
  hold on 
  contour(phi, [0,0], 'linewidth', 2.0, 'linecolor', 'r');
  hold off;
  
  title({'\bf Active contours without edges ', ... 
    ['$\mu$ = ', num2str(mu), ', C1 = ', num2str(C1, '%3.4g'), ...
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