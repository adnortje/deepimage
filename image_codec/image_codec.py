# imports 
import os
import numpy as np
from PIL import Image
from io import BytesIO
from skimage import measure
import torchvision.transforms.functional as F
from img_tools import ImageDataset, disp_images_widget, disp_prog_imgs


"""
ImageCodec
    
    Uses:
        Calculates Metric vs Bpp values for all images in img_dir given an image codec
    
        Supported Lossy Codecs:
            JPEG Progressive/ Non-Progressive
            JPEG 2000
            WEBP
            Note: should work with other codecs supported by PIL
            
        Supported Metrics:
            SSIM
            PSNR
    
        Args:
            img_dir : image directory
"""


class ImageCodec:
    
    def __init__(self, img_dir, codec, img_size=(224, 320)):
        
        img_dir = os.path.expanduser(img_dir)

        if not os.path.isdir(img_dir):
            raise NotADirectoryError("Specified directory d.n.e!")

        self.img_dir = img_dir
        
        # def image dataset
        self.img_set = ImageDataset(
            root_dir=self.img_dir
        )
        
        # def number of images
        self.img_no = len(self.img_set)
        
        # image codec
        if codec not in ['JPEG', 'WEBP', 'JPEG2000', 'JPEG (Progressive)']:
            raise ValueError("Specified codec is not currently supported!")

        self.codec_name = codec

        if codec in ['JPEG (Progressive)']:
            self.progressive = True
            codec = "JPEG"

        else:
            self.progressive = False

        self.codec = codec

        # def image size
        self.img_size = img_size
        
    def load_cc(self, metric, save_dir='./image_codec/saved_cc'):
        # load or create codec compression curve

        if not os.path.isdir(save_dir):
            raise ValueError("Specified directory d.n.e!")
        else:
            file_path = "".join(
                [save_dir, "/", self.codec_name, "_", metric, ".npy"]
            )
        
        try:
            cc = np.load(file_path).item()

        except FileNotFoundError:
            cc = self.get_cc(metric)
            np.save(file_path, cc)
        
        return cc
        
    def get_cc(self, metric):
        # calculate compression curve values

        met = []
        bpp = []
        
        for q in range(1, 95, 5):
            # for varying quantisation qualities
            m, b = self._avg_values(metric, q)
            met.append(m)
            bpp.append(b)
        
        cc = {'met': met, 'bpp': bpp}
        
        return cc

    def _encode_decode(self, r_img, q):
        # encode and decode with codec
        img_buff = BytesIO()

        if self.progressive:
            r_img.save(img_buff, format=self.codec, quality=q, progressive=True)
        else:
            r_img.save(img_buff, format=self.codec, quality=q)

        # calculate bpp
        w, h = r_img.size
        f_s = 8 * img_buff.getbuffer().nbytes
        bpp = f_s / (w * h)

        c_img = Image.open(img_buff).convert("RGB")

        # close image buffer
        img_buff.close()

        return c_img, bpp

    @staticmethod
    def _calc_ssim(r_img, c_img):
        # calculate SSIM
        ssim = measure.compare_ssim(
                    np.array(r_img),
                    np.array(c_img),
                    multichannel=True,
                    data_range=255,
                    gaussian_weights=True
        )
        return ssim

    @staticmethod
    def _calc_psnr(r_img, c_img):
        # calculate PSNR
        psnr = measure.compare_psnr(
                    np.array(r_img),
                    np.array(c_img),
                    data_range=255
        )
        return psnr
    
    def _avg_values(self, metric, q):
        # calculate average metric & bpp value pair for image dir
        
        run_met = 0.0
        run_bpp = 0.0
        
        for img in self.img_set:

            # convert to RGB and resize
            img = img.convert("RGB")
            img = F.resize(img, size=self.img_size)

            # encode image & calculate bpp
            c_img, bpp = self._encode_decode(img, q)
            run_bpp += bpp

            if metric == "PSNR":
                run_met += self._calc_psnr(img, c_img)
                
            elif metric == 'SSIM':
                run_met += self._calc_ssim(img, c_img)
        
        # calc and return averages    
        met = run_met / self.img_no
        bpp = run_bpp / self.img_no
        
        return met, bpp

    def find_nearest_bpp(self, img, target_bpp):
        # find nearest bpp encoding to target bpp
        best_img = None
        best_bpp = 1000

        for q in range(1, 95, 1):
            # encode image & calculate bpp
            c_img, bpp = self._encode_decode(img, q)

            if abs(target_bpp - bpp) < abs(target_bpp - best_bpp):
                # closer to target
                best_bpp = bpp
                best_img = c_img

                if best_bpp == target_bpp:
                    # break out once target bpp has been reached
                    break

        return best_img, best_bpp

    def progressive_imshow(self, itrs, widget=False):
        # display images at different stages of compression

        bpps = []
        ssim = []
        psnr = []
        imgs = []

        # load reference image
        r_img = self.img_set.__getitem__(0)
        r_img = r_img.convert("RGB")
        r_img = F.resize(r_img, size=self.img_size)

        for b in np.arange(0.125, itrs*0.125 + 0.125, 0.125):
            # locate nearest bpp value to b and encode
            c_img, bpp = self.find_nearest_bpp(r_img, b)

            # append compressed image and bpp
            bpps.append(round(bpp, 3))
            imgs.append(np.array(c_img))

            # append PSNR and SSIM
            psnr.append(
                round(self._calc_psnr(r_img, c_img), 2)
            )

            ssim.append(
                round(self._calc_ssim(r_img, c_img), 2)
            )

        if widget is True:
            # display image widget
            disp_images_widget(
                title = self.codec_name,
                bpp=bpps,
                imgs=imgs,
                ssim=ssim,
                psnr=psnr
            )

        else:
            # display images & bpp
            disp_prog_imgs(
                title=self.codec_name,
                bpp=bpps,
                imgs=imgs,
                ssim=ssim,
                psnr=psnr
            )

        return
