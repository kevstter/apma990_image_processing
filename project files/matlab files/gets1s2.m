function[S1,S2] = gets1s2(u, u0, thresh, S1, S2, eta, h)
%% Compute piecewise smooth region approx for fitting energy
%

  its = 0;
  
  tol = 1e-2;
  [M, N] = size(u0);
  a = h^2/eta;
  w = 1.25;
  
  oldS1 = zeros(size(S1));
%   oldS2 = zeros(size(S2));

  uzero = u<=thresh;
  
% Update pointwise -- SOR scheme; s-f = eta*laplacian(s)
while norm( oldS1 - S1, 'fro' ) > tol %|| norm( oldS2 - S2, 'fro' ) > tol
  its = its+1;
  oldS1 = S1; 
%   oldS2 = S2;
  for i = 2:M-1
    for j = 2:N-1
      c = [uzero(i+1,j) uzero(i-1,j) uzero(i,j+1) uzero(i,j-1)];
      if uzero(i,j) == 0 
        b = 4 - sum(c) + a;
        b = w/b; 
              
        S1(i,j) = (1-w)*S1(i,j) + b*( S1(i+1,j)*~c(1) + S1(i-1,j)*~c(2) ...
          + S1(i,j+1)*~c(3) + S1(i,j-1)*~c(4) + a*u0(i,j) );
        
        S2(i,j) = (1-w)*S2(i,j) + b*( S2(i+1,j)*~c(1) + S2(i-1,j)*~c(2) ...
          + S2(i,j+1)*~c(3) + S2(i,j-1)*~c(4) + a*u0(i,j) );   
      else
        b = sum(c) + a;
        b = w/b;
      
        S1(i,j) = (1-w)*S1(i,j) + b*( S1(i+1,j)*c(1) + S1(i-1,j)*c(2) ...
          + S1(i,j+1)*c(3) + S1(i,j-1)*c(4) + a*u0(i,j) );  
        
        S2(i,j) = (1-w)*S2(i,j) + b*( S2(i+1,j)*c(1) + S2(i-1,j)*c(2) ...
          + S2(i,j+1)*c(3) + S2(i,j-1)*c(4) + a*u0(i,j) );   
      end
    end
  end
end

  fprintf('%d, ', its);
  
% Update BC: BC at the interface and BC at the edges of image domain
% 1. Find interface
%   ij = find_interface(u, thresh, N);
  
% 2. Advect in normal dir for a few time steps to satisfy BC @ interface
  oldS1 = S1;
  oldS2 = S2;
  [S1, S2] = extend_vel(u, thresh, S1, S2, oldS1, oldS2, h);
   
% 3: Set BC at the edges of image domain
  S1 = BCs(S1, M, N);
  S2 = BCs(S2, M, N);
  
end

function[S1, S2] = extend_vel(u, thresh, S1, S2, oldS1, oldS2, h)
%% Velocity extension to enforce boundary condition grad(S) dot n = 0;
% See 'velocity extension', 'ghost fluid method'
%

% Time parameters
  dt = 0.50*h; 
  tf = 10; 
  nt = ceil(tf/dt); 
  dt = tf/nt;
  ep = 1e-3;
  
% Arrays for Neumann boundary conditions
  [M, N] = size(u);
  ip = 2:N+1; ip(N) = N;
  im = 0:N-1; im(1) = 1;

% Set advection speeds for extension, Godunov upwind scheme
% normal in x-dir, n_x
  uxm = (u-u(im,:))/h; 
  uxp = (u(ip,:)-u)/h;
  uxmp = max(uxm,0); 
  uxpm = min(uxp,0);
  gx = abs(uxmp)>abs(uxpm);
  dgx = uxmp.*gx + uxpm.*~gx;
  
% normal in y-dir, n_y  
  uym = (u-u(:,im))/h; 
  uyp = (u(:,ip)-u)/h;
  uymp = max(uym,0); 
  uypm = min(uyp,0);
  gy = abs(uymp)>abs(uypm);
  dgy = uymp.*gy + uypm.*~gy;

% Smoothed denominator  
  agrad = sqrt(dgx.^2+dgy.^2+ep^2);
  
% Grouping terms that form the 'advection speed' and determine upwind dir
  nx1 = dgx./agrad; 
  ny1 = dgy./agrad;
  nxp1 = nx1.*(nx1>0); 
  nyp1 = ny1.*(ny1>0); 
  nxm1 = nx1.*(nx1<=0); 
  nym1 = ny1.*(ny1<=0);
  
  nx2 = -dgx./agrad; 
  ny2 = -dgy./agrad;
  nxp2 = nx2.*(nx2>0); 
  nyp2 = ny2.*(ny2>0); 
  nxm2 = nx2.*(nx2<=0); 
  nym2 = ny2.*(ny2<=0);
  
  cond1 = u<thresh;
  cond2 = u>=thresh;
  
  for iter=1:nt
 % Extend S1    
    dmx = S1-S1(im,:);  % use with nx>0
    dmy = S1-S1(:,im);  % use with ny>0
    dpx = S1(ip,:)-S1;  % use with nx<0
    dpy = S1(:,ip)-S1;  % use with ny<0
    ndx = dmx.*nxp1 + dpx.*nxm1; 
    ndy = dmy.*nyp1 + dpy.*nym1;
    S1 = S1-(dt/h)*(ndx+ndy);
    
    S1(cond1) = oldS1(cond1);
    
  % Extend S2
    dmx = S2-S2(im,:);  % use with nx>0
    dmy = S2-S2(:,im);  % use with ny>0
    dpx = S2(ip,:)-S2;  % use with nx<0
    dpy = S2(:,ip)-S2;  % use with ny<0
    ndx = dmx.*nxp2 + dpx.*nxm2; 
    ndy = dmy.*nyp2 + dpy.*nym2;
    S2 = S2-(dt/h)*(ndx+ndy);
    
    S2(cond2) = oldS2(cond2);
    
  end

end

function[ij] = find_interface(u, thresh, N)
%% Find grid points next to interface defined by the level set of u:
% {x : u(x) > thresh}
%
% Returns pts on both sides of the interface. Half should be on the side 
% u(x)>thresh, half on the side u(x)<thresh
%

% Process contour C from contourc -- C may be composed of multiple closed
% contours. Each closed contour is preceded by an entry in cc giving its 
% length. For example, cc(:,1) = (0, 120) means the  coordinates 
% cc(:,2:121) are points on the first closed contour.
  cc = contourc(u, [thresh thresh]);  
  posn = 1;
  while posn <= length(cc)
    temp = posn;
    posn = posn + cc(2, posn);
    cc(:,temp) = [];  
  end

% Find grid pts next to the contour. Points from cc are interpolated; they 
% are, in general, not grid points
  interface_pts = zeros(2,2*length(cc));
  posn = 1;
  for k=1:length(cc)
    cc1 = cc(1,k);
    cc2 = cc(2,k);
    if cc1 ~= floor(cc1)
      x1 = ceil(cc1);
      interface_pts(:, posn) = [x1; cc2];
      interface_pts(:, posn+1) = [x1-1; cc2];
    else
      y1 = ceil(cc2);
      interface_pts(:, posn) = [cc1; y1];
      interface_pts(:, posn+1) = [cc1; y1-1];
    end
    posn = posn + 2;
  end

  jj = interface_pts(1,:); 
  ii = interface_pts(2,:);  
  ij = (jj-1)*N + ii;
%   ij = ij( phi(ij)<0 );
  ij = unique( ij );
end

