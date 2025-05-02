% Implementation of the Wirtinger Flow (WF) algorithm presented in the paper 
% "Phase Retrieval via Wirtinger Flow: Theory and Algorithms" 
% by E. J. Candes, X. Li, and M. Soltanolkotabi

% The input data are coded diffraction patterns about a random complex
% valued image. 

n1 = 128;
n2 = 256;
x = randn(n1,n2) + 1i*randn(n1,n2);
img = imread('/Users/yangtaewon/Desktop/00000.png');
img = rgb2gray(img);            
img = imresize(img, [n1, n2]);
img = double(img);
img = img - mean(img(:));
img = img / std(img(:));

%% Apply Fourier Transform to make it complex
x = fft2(img); % Fourier-transformed image as complex input
%% Make masks and linear sampling operators

L = 5;                  % Number of masks  
Masks = zeros(n1,n2,L);  % Storage for L masks, each of dim n1 x n2

% Sample phases: each symbol in alphabet {1, -1, i , -i} has equal prob. 
for ll = 1:L, Masks(:,:,ll) = randsrc(n1,n2,[1i -1i 1 -1]); end

% Sample magnitudes and make masks 
temp = rand(size(Masks));
Masks = Masks .* ( (temp <= 0.2)*sqrt(3) + (temp > 0.2)/sqrt(2) );

% Make linear operators; A is forward map and At its scaled adjoint (At(Y)*numel(Y) is the adjoint) 
A = @(I)  fft2(conj(Masks) .* reshape(repmat(I,[1 L]), size(I,1), size(I,2), L));  % Input is n1 x n2 image, output is n1 x n2 x L array
At = @(Y) mean(Masks .* ifft2(Y), 3);                                              % Input is n1 x n2 X L array, output is n1 x n2 image

% Data 
Y = abs(A(x)).^2;  

%% Initialization

npower_iter = 50;                          % Number of power iterations 
z0 = randn(n1,n2); z0 = z0/norm(z0,'fro'); % Initial guess 
tic                                        % Power iterations 
for tt = 1:npower_iter, 
    z0 = At(Y.*A(z0)); z0 = z0/norm(z0,'fro');
end
Times = toc; 

normest = sqrt(sum(Y(:))/numel(Y)); % Estimate norm to scale eigenvector  
z = normest * z0;                   % Apply scaling 
Relerrs = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error

%% Loop

T = 500;                            % Max number of iterations
tau0 = 330;                         % Time constant for step size
mu = @(t) min(1-exp(-t/tau0), 0.4); % Schedule for step size

tic
for t = 1:T
    Bz = A(z);
    C  = (abs(Bz).^2-Y) .* Bz;
    grad = At(C);                    % Wirtinger gradient
    z = z - mu(t)/normest^2 * grad;  % Gradient update 
    
    Relerrs = [Relerrs, norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro')]; 
    Times  = [Times, toc] ;   
end
fprintf('All done!\n')

%%  Check results

 fprintf('Run time of initialization: %.2f\n', Times(1))
 fprintf('Relative error after initialization: %f\n', Relerrs(1))
 fprintf('\n')
 fprintf('Loop run time: %.2f\n', Times(T+1))
 fprintf('Relative error after %d iterations: %f\n', T, Relerrs(T+1))
 
 figure, semilogy(0:T,Relerrs) 
 xlabel('Iteration'), ylabel('Relative error (log10)'), ...
     title('Relative error vs. iteration count')
 figure;
 subplot(1,2,1);
 imagesc(real(z));
 subplot(1,2,2);
 imagesc(real(x));

 % Recover the real-world image from the reconstructed complex estimate
 reconstructed_image = abs(ifft2(z));  % Inverse FFT to reconstruct the image 
 original_image = abs(ifft2(x));       % Reference original image

 % Display the original and reconstructed images
 figure;
 subplot(1,2,1);
 imagesc(original_image); colormap gray; axis off;
 title('Original Image')

 subplot(1,2,2);
 imagesc(reconstructed_image); colormap gray; axis off;
 title('Reconstructed Image')

rec_norm = mat2gray(reconstructed_image);             
rec_uint8 = im2uint8(rec_norm);                       

imwrite(rec_uint8, 'reconstructed_image.png');        

orig_norm = mat2gray(original_image);
orig_uint8 = im2uint8(orig_norm);
imwrite(orig_uint8, 'original_image.png');
