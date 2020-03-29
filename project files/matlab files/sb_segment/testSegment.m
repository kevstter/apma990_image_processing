%% testSegment.m
%
% This code tests the "sbseg" method.  The function "sbseg" must be
% compiled separately before this method is called.
%
% To compile in matlab command line
% >> mex sbseg.c
%
% References: 
% [1] GoldsteinBressonOsher(2010), Geometric applications of the split ...
%
% Retrieved: March2020, http://tag7.web.rice.edu/Split_Bregman.html
% Last edited: 29Mar2020, KChow
%

%% Basic testing
% Series of synthetic images avail for testing
im = {'orig','sqr','sqr2','sqr3','blur','blur2','sidebar','bar'};

% close all; clear;
for k=1:1%length(im)
  u0 = get_im(im{k});

% Add noise
  u0 = u0+10*randn(256,256);

%  Edge detector
  edge = ones(256,256);     % do nothing edge detector
%   edge = imgaussfilt( image );  % Smooth out image
%   edge = 1./(1 + imgradient( edge ).^2); % classic edge detector, p=2

% Segment with 3 different parameters
  m1 = 1e-2; m2 = 1e-5; m3 = 2e-6;
  u1 = sbseg(u0, edge, m1);
  u2 = sbseg(u0, edge, m2);
  u3 = sbseg(u0, edge, m3);

% show results
  plot_results(u0, u1, u2, u3, m1, m2, m3)

end
% End of testing
  
function[] = plot_results(u0, u1, u2, u3, m1, m2, m3)
  thres = 0.5;
  figure; 
  subplot(2,2,1);
  imagesc(u0); axis image
  title('original image');

  subplot(2,2,2);
  imagesc(u1>thres); axis image
  title(['mu=',num2str(m1)]);

  subplot(2,2,3);
  imagesc(u2>thres); axis image
  title(['mu=',num2str(m2)]);

  subplot(2,2,4);
  imagesc(u3>thres); axis image
  title(['mu=',num2str(m3)]);
end

function[u] = get_im( im )
  u = zeros(256,256);
  switch im
    case 'orig'
      u(10,:) = 256;          % adds a line
      u(50:100,50:100) = 256; % adds a rectangle
      u(150:170,150:170)=256; % adds second rectangle
      
    case 'sqr' % centred square
      u(100:150,100:150) = 255; 
      
    case 'sqr2' % L-shape
      u(78:178,78:128) = 255;
      u(129:178,129:179) = 255; 
      
    case 'sqr3' % L-shape + small block
      u(78:178, 78:128) = 255;
      u(129:178, 129:179) = 255; 
      u(78:118, 139:179) = 255;
      
    case 'blur' % gaussian bump; blurred circle
      xx = 1:256;
      [XX, YY] = meshgrid(xx, xx);
      D = sqrt((XX-128).^2 + (YY-128).^2);
      u( D < 25 ) = 256;
      u = imgaussfilt( u, 15 ); 
      
    case 'blur2' % gaussian bump; blurred circle
      xx = 1:256;
      [XX, YY] = meshgrid(xx, xx);
      D = sqrt((XX-128).^2 + (YY-128).^2);
      u( D < 25 ) = 255;
      u = imgaussfilt( u, 40 ); 
      
    case 'sidebar'
      u(96:160, 112:128) = 255;
      for i = 129:192
        u(96:160, i) = -1/16*(i-128).^2 + 255;
      end
      u = circshift(u, [0,-111]);
      
    case 'bar'
      u(96:160, 64:128) = 255;
      ep = 1e-1;
      p = 2;
      a = (ep - 4^4)/(4^(3/p));
      for i = 129:192
        u(96:160, i) = a*(i-128)^(1/p) + 255;
      end
  end
end