# imports
import os
import torch
import numpy as np
import img_tools as im_t
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from image_codec import ImageCodec
from torchvision.utils import make_grid
from img_tools import EvaluationImageDataLoaders, InvNormalization


# -----------------------------------------------------------------------------------------------------------------------
# Comparison of Models Class
# -----------------------------------------------------------------------------------------------------------------------

# constant variables
SAVED_MODELS_PATH = "./saved_models/"
SAVED_CODECS_PATH = "./image_codec/saved_cc/"

"""
CompareModels
 
 Compares performance various Autoencoder Models
         
         Args:
            img_dir (String)             : path to image directory
            models  (list of nn.Modules) : list of trained models to compare
            codecs  (list of String)     : list of standard lossy codec names 
            dataset (string)             : dataset on which to perform evaluation ['test', 'valid']

"""


class CompareModels:
    
    def __init__(self, img_dir, models, codecs):
        
        # Image Directory
        img_dir = os.path.expanduser(img_dir)

        if os.path.isdir(img_dir):
            self.img_dir = img_dir
        else:
            raise NotADirectoryError(
                img_dir + " is not a directory"
            )

        # List Standard Lossy Codecs
        self.codecs = codecs

        # List Deep Image Compression Models
        self.models = models

    def display_compression_curve(self, metric, dataset='valid'):

        # plot compression curves
        
        # setup plot
        im_t.setup_plot(
            title='',
            y_label=metric,
            x_label="bits-per-pixel (bpp)"
        ) 
        
        legend = []
        
        fmt_index = 0
        fmts = ['^--b', '*--c', 'o--g', 'x--r', 'D--r']
        
        # plot for Deep Compression Models

        for model in self.models:
            
            legend.append(model.display_name)
            
            # load compression curve
            curve = EvaluateModel(self.img_dir, model).load_compression_curve(
                save_dir=SAVED_MODELS_PATH + model.name + '/' + dataset,
                metric=metric
            )

            # plot rate-distortion curve
            plt.plot(curve['bpp'], curve['met'], fmts[fmt_index])
            fmt_index += 1
        
        # plot for Standard Image Codecs
        for codec in self.codecs:
            
            legend.append(codec)
            
            # load compression curve
            curve = ImageCodec(self.img_dir, codec).load_cc(
                metric=metric,
                save_dir=SAVED_CODECS_PATH + dataset
            )

            # plot rate distortion curve
            plt.plot(curve['bpp'], curve['met'], fmts[fmt_index])
            fmt_index += 1
        
        plt.legend(legend, loc='lower right', fontsize='large')
        plt.savefig('./'+metric+'_curve.pdf')
        plt.show()
        
        return
    
    def display_compressed_images(self, itrs, dataset='valid'):

        # plot progressive images

        for model in self.models:
            EvaluateModel(
                self.img_dir+"/"+dataset,
                model
            ).progressive_imshow(itrs)

        for codec in self.codecs:

            ImageCodec(
                self.img_dir+"/"+dataset,
                codec
            ).progressive_imshow(itrs)
        
        return

    def display_auc(self, metric, dataset, bpp_max=2.0, bpp_min=0.0):

        # display Area Under Curve
        print("Displaying AUC:")
        print("\nDeep Compression Models")

        for model in self.models:

            # load compression curve
            curve = EvaluateModel(self.img_dir, model).load_compression_curve(
                metric=metric,
                save_dir=SAVED_MODELS_PATH + model.name + '/' + dataset
            )

            cut_curve = {'bpp': [], 'met': []}
            for i in range(len(curve['bpp'])):
                if bpp_min <= curve['bpp'][i] <= bpp_max:
                    # save values in range
                    cut_curve['bpp'].append(curve['bpp'][i])
                    cut_curve['met'].append(curve['met'][i])

            curve_area = auc(cut_curve['bpp'], cut_curve['met'])

            print("{} : {}".format(model.display_name, curve_area))

        print("\nStandard Codecs")

        # plot compression curves for standard image codecs
        for codec in self.codecs:

            # load compression curve
            curve = ImageCodec(self.img_dir, codec).load_cc(
                metric=metric,
                save_dir=SAVED_CODECS_PATH + dataset
            )

            cut_curve = {'bpp': [], 'met': []}

            for i in range(len(curve['bpp'])):

                if bpp_min <= curve['bpp'][i] <= bpp_max:
                    # save values in range
                    cut_curve['bpp'].append(curve['bpp'][i])
                    cut_curve['met'].append(curve['met'][i])

            # interpolate last point
            cut_curve['met'].append(
                (curve['met'][-1] - cut_curve['met'][-1])/curve['bpp'][-1] * (2.0 - cut_curve['bpp'][-1]) + cut_curve['met'][-1]
            )
            cut_curve['bpp'].append(2.0)

            curve_area = auc(cut_curve['bpp'], cut_curve['met'])

            print("{} : {}".format(codec, curve_area))

        return

    
# ----------------------------------------------------------------------------------------------------------------------
# Single Model Evaluation Class
# ----------------------------------------------------------------------------------------------------------------------

"""
Class EvaluateModel:

    Various methods used to evaluate an Autoencoder model (full iteration evaluations)
    
        Args:
            model (nn.Module) : trained model to be evaluated
"""


class EvaluateModel:
    
    def __init__(self,
                 img_dir,
                 model, img_s=(224, 320)):
        
        # use GPU if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
     
        self.model = model.to(self.device)

        # run model in inference mode
        self.model.train(False)

        if model.name in ['BINetAR', 'BINetOSR', 'BINetOSR1']:
            self.p_s = self.model.g_s
        else:
            self.p_s = self.model.p_s

        img_dir = os.path.expanduser(img_dir)

        if os.path.isdir(img_dir):
            self.img_dir = img_dir
        else:
            raise NotADirectoryError

        # define Evaluation dataloader
        eval_dls = EvaluationImageDataLoaders(
            img_dir=img_dir,
            img_s=img_s,
            p_s=self.p_s,
            b_s=1
        )

        # Inverse Normalization Transform [-1, 1] -> [0, 1]
        self.inv_norm = InvNormalization(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
        
        # Image DataLoader
        self.img_dl = eval_dls.get_img_dl()
        
        # Patch DataLoader
        eval_dls.b_s = 5
        self.patch_dl = eval_dls.get_patch_dl()
                
    def _progressive_eval(self, metric):

        # calculate average quality at each model itr

        # metric values
        m_val = []
        
        for i in range(self.model.itrs):
            
            # calculate & append metric value
            m_val.append(
                self._average_eval(metric, i+1)
            )
            
        return m_val
        
    def _average_eval(self, metric, itrs, print_stat=False):
        
        # calculate average quality at specific itr

        # init variables
        m_val = []

        if metric not in ['SSIM', 'PSNR']:
            raise KeyError('Unsupported Metric : ' + metric)

        for r_img in self.img_dl:

            # encode & decode images
            c_img = self.model.encode_decode(
                r_img.to(self.device),
                itrs
            )

            # [-1, 1] -> [0, 1]
            r_img = self.inv_norm(r_img[0])
            c_img = self.inv_norm(c_img[0].cpu())

            # evaluate metric
            if metric == 'SSIM':
                m_val.append(
                    self._evaluate_ssim(r_img, c_img)
                )

            elif metric == 'PSNR':
                m_val.append(
                    self._evaluate_psnr(r_img, c_img)
                )
                
        # average metric value
        m_avg = sum(m_val) / len(m_val)
        
        if print_stat:
            # print average metric
            print(
                "Average " + metric + " over {} Images:".format(len(m_val))
            )
            print(str(m_avg))
     
        return m_avg
    
    def _calc_compression_curve(self, metric, save_dir):

        # calculate and save rate-distortion curve
        
        # bpp axis
        bpp = self.model.b_n / (self.model.p_w * self.model.p_h)
        bpp = np.linspace(
            bpp, bpp * self.model.itrs, self.model.itrs
        )
        
        # calculate metric values
        m_val = self._progressive_eval(metric)
        
        # compression curve dictionary
        curve = {
            'bpp': bpp,
            'met': m_val
        }
        
        # create file name
        file_name = "".join([
            save_dir,
            '/',
            self.model.name,
            '_',
            metric,
            '.npy'
        ])

        # save curve as numpy file
        np.save(file_name, curve)
 
        return
        
    def load_compression_curve(self, metric, save_dir):

        if metric not in ['SSIM', 'PSNR']:
            raise KeyError('Unsupported Metric : ' + metric)

        save_dir = os.path.expanduser(save_dir)

        if not os.path.isdir(save_dir):
            raise NotADirectoryError("Specified directory does not exist!")
        
        # create filename
        file_name = "".join([
            save_dir,
            '/',
            self.model.name,
            '_',
            metric,
            '.npy'
        ])
        
        if not os.path.isfile(file_name):

            print('Creating Compression Curve File : ')
            
            # calculate compression curve
            self._calc_compression_curve(
                metric=metric,
                save_dir=save_dir
            )

        # load curve
        curve = np.load(file_name).item()
            
        return curve
    
    def compare_img(self, itrs):
        
        # display reference and compressed image side by side

        # get image
        r_img = iter(self.img_dl).next()
        
        # encode & decode image
        c_img = self.model.encode_decode(
            r_img.to(self.device),
            itrs
        )

        # Inverse Normalization [-1,1] -> [0,1]
        r_img = self.inv_norm(r_img[0])
        c_img = self.inv_norm(c_img[0].cpu())

        # display images
        self._display_images(r_img, c_img)

        return
    
    def save_img(self, itrs):
        
        # get image
        r_img = iter(self.img_dl).next()
        
        # encode & decode image
        c_img = self.model.encode_decode(
            r_img.to(self.device),
            itrs
        )

        # Inverse Normalization [-1,1] -> [0,1]
        c_img = self.inv_norm(c_img[0].cpu())
        
        # save image
        im_t.save_img(c_img)
        
        return
    
    def compare_patches(self, itrs):
        
        # display reference & compressed patches side by side
        
        # fetch patches
        r_patches = iter(self.patch_dl).next()

        # encode & decode patches

        c_patches = self.model.encode_decode(
            r_patches.to(self.device),
            itrs
        ).cpu()
        
        r_patches = make_grid(
            r_patches,
            nrow=self.patch_dl.batch_size,
            padding=0
        )

        c_patches = make_grid(
            c_patches,
            nrow=self.patch_dl.batch_size,
            padding=0
        )

        # [-1,1] -> [0,1]
        r_patches = self.inv_norm(r_patches)
        c_patches = self.inv_norm(c_patches)

        # display patches
        self._display_images(r_patches, c_patches)

        return

    @staticmethod
    def _print_evaluation(ref, comp):
        
        # prints evaluation metrics
        
        # calculate & display SSIM
        ssim = im_t.EvalMetrics.SSIM.calc(
            ref, 
            comp
        )
        print("SSIM : {}".format(ssim))
        
        # calculate & display PSNR
        psnr = im_t.EvalMetrics.PSNR.calc(
            ref, 
            comp
        )
        print("PSNR : {}".format(psnr))
        
        return

    @staticmethod
    def _evaluate_ssim(ref, comp):
        
        # calculate SSIM
        ssim = im_t.EvalMetrics.SSIM.calc(
            ref, 
            comp
        )
        
        return ssim

    @staticmethod
    def _evaluate_psnr(ref, comp):
        
        # calculate PSNR
        psnr = im_t.EvalMetrics.PSNR.calc(
            ref, 
            comp
        )
        
        return psnr

    def _display_images(self, ref, comp):

        # display reference & compressed images
        im_t.vs_imshow(ref, comp)
        
        # display compression metrics
        self._print_evaluation(ref, comp)
        
        return
    
    def progressive_imshow(self, itrs, widget=False):
        
        # display images at different stages of compression
        
        bpp = []
        ssim = []
        psnr = []
        c_imgs = []
        
        # get input image
        r_img = iter(self.img_dl).next()

        for i in range(itrs):
            
            # append bpp
            bits = self.model.b_n * (i+1) / (self.model.p_w * self.model.p_h)
            bpp.append(bits)

            # encode & decode image
            c_img = self.model.encode_decode(
                r_img.to(self.device),
                i+1
            ).cpu()

            # [-1,1] -> [0,1]
            c_img = self.inv_norm(c_img[0])
            c_imgs.append(c_img)

        # [-1,1] -> [0,1]
        r_img = self.inv_norm(r_img[0])
        c_imgs = torch.stack(c_imgs, dim=0).cpu()

        for c_img in c_imgs:
            # calculate SSIM & PSNR
            ssim.append(
                round(self._evaluate_ssim(r_img, c_img), 2)
            )

            psnr.append(
                round(self._evaluate_psnr(r_img, c_img), 2)
            )
        
        if widget is True:
            # display widget
            im_t.disp_images_widget(
                title=self.model.display_name,
                imgs=c_imgs,
                bpp=bpp,
                ssim=ssim,
                psnr=psnr
            )
            
        else:
            # display images & bpp
            im_t.disp_prog_imgs(
                title=self.model.display_name,
                imgs=c_imgs,
                bpp=bpp,
                ssim=ssim,
                psnr=psnr
            )
        
        return

