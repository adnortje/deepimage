import numpy as np
from skimage import measure


class EvalMetrics:
    
    class SSIM:
        """
        Method: calc
    
            Function used to compute the M-SSIM between two Tensor images
        
                Args:
                    r_img (torch.Tensor): reference or original images (C,H,W)
                    c_img (torch.Tensor): compressed image (C,H,W)
                Returns:
                    ssim (float): comparative SSIM between the two images
        """
        @staticmethod
        def calc(r_img, c_img):
            
            # to CPU for np
            r_img = r_img.cpu()
            c_img = c_img.cpu()
            
            # Tensor -> Numpy images [h, w, C]
            r_img = np.transpose(r_img.numpy(), (1, 2, 0)) * 255
            c_img = np.transpose(c_img.numpy(), (1, 2, 0)) * 255

            ssim = measure.compare_ssim(
                r_img, 
                c_img,
                data_range=255,
                multichannel=True,
                gaussian_weights=True
            )
            return ssim

        @staticmethod
        def check_threshold(thresh):
            assert(-1.0 <= thresh <= 1.0), "SSIM Threshold out of bounds, [-1,1]"
            return
    
    class PSNR:
        """
        Method: calc
        
            Function used to compute the PSNR between two Tensor images
               
                Args:
                    r_img (torch.Tensor): reference or original image (C, H, W)
                    c_img (torch.Tensor): compressed image (C, H, W)
                Returns:
                    psnr (float): comparative M-SSIM between the two images
        """
        @staticmethod
        def calc(r_img, c_img):
            
            # to CPU for np
            r_img = r_img.cpu()
            c_img = c_img.cpu()
            
            # Tensor -> Numpy images [h, w, C]
            r_img = np.transpose(r_img.numpy(), (1, 2, 0)) * 255
            c_img = np.transpose(c_img.numpy(), (1, 2, 0)) * 255

            psnr = measure.compare_psnr(
                r_img, 
                c_img,
                data_range=255
            )
            return psnr

        @staticmethod
        def check_threshold(thresh):
            assert(0.0 <= thresh), "PSNR Threshold should not be Negative"
            return

