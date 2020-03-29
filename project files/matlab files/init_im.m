function[u0, r] = init_im( im )
%% Create simple synthetic image for testing segmentation codes
  u0 = zeros(256,256);
  switch im
    case 'sqr1' % centred square
      u0(100:150,100:150) = 255; 
      r = 45;
      
    case 'sqr2' % offset squares
      u0(50:100,50:100) = 255; 
      u0(150:170,150:170) = 255;
      r = 1;
      
    case 'sqr3' % L-shape
      u0(78:178,78:128) = 255;
      u0(129:178,129:179) = 255; 
      r = 75;
      
    case 'sqr4' % L-shape + small block
      u0(78:178, 78:128) = 255;
      u0(129:178, 129:179) = 255; 
      u0(78:118, 139:179) = 255;
      r = 85;
      
    case 'blur' % gaussian bump; blurred circle
      xx = 1:256;
      [XX, YY] = meshgrid(xx, xx);
      D = sqrt((XX-128).^2 + (YY-128).^2);
      u0( D < 25 ) = 256;
      u0 = imgaussfilt( u0, 15 ); 
      r = 125;
      
    case 'blur2' % gaussian bump; blurred circle
      xx = 1:256;
      [XX, YY] = meshgrid(xx, xx);
      D = sqrt((XX-128).^2 + (YY-128).^2);
      u0( D < 25 ) = 255;
      u0 = imgaussfilt( u0, 40 ); 
      r = 125;
      
    case 'sidebar'
      u0(96:160, 112:128) = 255;
      for i = 129:192
        u0(96:160, i) = -1/16*(i-128).^2 + 255;
      end
      u0 = circshift(u0, [0,-111]);
      r = 111;
      
    case 'bar'
      u0(96:160, 64:128) = 255;
      epsl = 1e-1;
      p = 2;
      a = (epsl - 4^4)/(4^(3/p));
      for i = 129:192
        u0(96:160, i) = a*(i-128)^(1/p) + 255;
      end
      r = 75;
    
    otherwise
      fprintf('\nDefault u0 -- possibly undefined behaviour.\n');
      u0 = im2double(imread('cameraman.tif'));
      r = min(size(u0))/2;
      
  end
end