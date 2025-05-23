from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class FFTProcessor():
    
    '''signal to FFT image'''
    
    def __init__(self, opt):
        self.opt = opt
        #self.random_seed = random_seed
        
    def generate_random_signal(self, length):
        t = np.linspace(0, length, int(self.opt.sampling_frequency), endpoint=False)
        x = np.zeros_like(t)
        
        num_components = np.random.randint(20, 50)  # 2~4
        for _ in range(num_components):
            A = np.random.uniform(0.5, 3.0)            # 진폭
            f = np.random.uniform(1, 20)               # 주파수 (Hz)
            phi = np.random.uniform(0, 2*np.pi)        # 위상
            x += A * np.sin(2 * np.pi * f * t + phi)
        return x

    def process_batch(self, img_x_dir, img_y_dir):
        for sample_idx in range(self.opt.n_samples):
            #np.random.seed(self.random_seed)
            x = self.generate_random_signal(self.opt.lr_length)  # 이렇게 opt. 쓰는 것?? 별론가
            y = self.generate_random_signal(self.opt.hr_length)

            X = np.fft.fft(x)
            Y = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(X), d=1/self.opt.sampling_frequency)
            mag_X = np.abs(X)* 2 / len(X)
            mag_Y = np.abs(Y)* 2 / len(Y)

        
            half = int(len(X)/2)
            freq_mag_X = np.column_stack((freqs[:half], mag_X[:half]))
            freq_mag_Y = np.column_stack((freqs[:half], mag_Y[:half]))
            
            save_img(freq_mag_X, img_x_dir, sample_idx)
            save_img(freq_mag_Y, img_y_dir, sample_idx)
            
            save_patch(f"{img_x_dir}/fft_{sample_idx:03d}.png", img_x_dir, sample_idx, self.opt.patch_size, self.opt.stride)
            save_patch(f"{img_y_dir}/fft_{sample_idx:03d}.png", img_y_dir, sample_idx, self.opt.patch_size, self.opt.stride)


def save_img(x, img_dir, idx):
    
    plt.plot(x[:, 0], x[:, 1])
    plt.axis('off')     # 축 숨기기 (x, y 모두)
    plt.gca().spines['top'].set_visible(False)    # 테두리(스파인) 숨기기
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.savefig(f"{img_dir}/fft_{idx:03d}.png", bbox_inches = 'tight', transparent = True)
    plt.close()
    
def save_patch(x, img_dir, sample_idx, patch_size, stride):
    patch_idx = 0
    img = Image.open(x)
    img = np.array(img)
    for i in range(0, img.shape[0] - patch_size + 1, stride):
        for j in range(0, img.shape[1] - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            plt.imsave(f"{img_dir}/fft_{sample_idx:03d}_{patch_idx:02d}.png", patch, cmap='gray')
            patch_idx += 1