from PIL import Image, ImageDraw
import numpy as np
from scipy import fftpack
import sys

def image_to_np_array(image_path):
    return np.asarray(Image.open(image_path).convert('L'))

def resize_image(image_np_array, shape):
    PIL_image = Image.fromarray(np.uint8(image_np_array)).convert("L")
    PIL_image = PIL_image.resize(shape)
    return np.array(PIL_image)

def fourier_transform(image_np):
    return fftpack.fftshift(fftpack.fft2(image_np)) # np.fft.fftshift(np.fft.fft2(np.float32(image_np)))

def inverse_fourier_transform(fft_image_np):
    return abs(fftpack.ifft2(fftpack.ifftshift(fft_image_np)))

def save_fft_image(image_fourier_np, filename):
    fourier_image_np = (np.log(abs(image_fourier_np))* 255 /np.amax(np.log(abs(image_fourier_np)))).astype(np.uint8)
    fourier_image = Image.fromarray(fourier_image_np).convert("L")
    fourier_image.save(filename)


def fft_pass_filter(fft_image_np, is_high = False, radius = 30):
    eX, eY = radius, radius
    x, y = fft_image_np.shape[1], fft_image_np.shape[0]
    bounding_box = (x/2 - eX/2, y/2 - eY/2, x/2 + eX/2, y/2 + eY/2)
    
    color = 255 if not is_high else 0
    fill  = 0 if not is_high else 255

    filter = Image.new("L", (x, y), color=color) 
    draw = ImageDraw.Draw(filter)
    draw.ellipse(bounding_box, fill=fill)
    return np.array(filter)


def fft_join_image(fft_input_np, image_1_np, image_2_np):
    final_image = np.full(image_1_np.shape, 255, dtype = complex)
    for i in range(image_1_np.shape[0]):
        for j in range(image_1_np.shape[1]):
            if fft_input_np[i,j] == 0: 
                final_image[i,j] = image_1_np[i,j]
            else: 
                final_image[i,j] = image_2_np[i,j]
    return final_image


if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print("The arguments must be as follows:\npython3 lab1.py <<image_1_path>> <<image_2_path>>")
        sys.exit()

    sample_gates = image_to_np_array(sys.argv[1])
    clock = image_to_np_array(sys.argv[2])

    sample_gates_cropped = resize_image(sample_gates, (600, 800))
    clock_resized = resize_image(clock, (600, 800))


    fft_clock = fourier_transform(clock_resized)
    fft_sample_gates = fourier_transform(sample_gates_cropped)

    save_fft_image(fft_sample_gates, "sample_gates_fft.png")
    save_fft_image(fft_clock, "clock_fft.png")

    fft_clock_low = fft_pass_filter(fft_clock)
    fft_clock_high = fft_pass_filter(fft_clock, True)

    fft_sg_low = fft_pass_filter(fft_sample_gates)
    fft_sg_high = fft_pass_filter(fft_sample_gates, True)

    fft_clock_low_sg_high = fft_join_image(fft_clock_low, fft_clock, fft_sample_gates)
    fft_clock_high_sg_low = fft_join_image(fft_clock_high, fft_clock, fft_sample_gates)

    clock_low_sg_high = inverse_fourier_transform(fft_clock_low_sg_high)
    clock_high_sg_low = inverse_fourier_transform(fft_clock_high_sg_low)

    clock_low_sg_high = Image.fromarray(clock_low_sg_high).convert("L")
    clock_low_sg_high.save("Image_1_fft_ifft.jpg")

    clock_high_sg_low = Image.fromarray(clock_high_sg_low).convert("L")
    clock_high_sg_low.save("Image_2_fft_ifft.jpg")