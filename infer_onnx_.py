import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

def get_color_map_list(num_classes, custom_color=None):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map
def preprocess(img_path, input_size=(1024,1024), mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))[None, ...]  # [1, C, H, W]
    return img

def infer_onnx(img_path, onnx_path, save_path):
    img = preprocess(img_path)
    sess = ort.InferenceSession(onnx_path)
    img = img.astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: img})[0]  # [1, C, H, W]

    pred = np.argmax(out[0], axis=0).astype(np.uint8)  # [H, W]
    mask = Image.fromarray(pred, mode='P')
    palette = get_color_map_list(256)
    mask.putpalette(palette)
    mask.save(save_path)
    print(f"Saved prediction: {save_path}")

if __name__ == '__main__':
    img_path = 'test.jpg'
    onnx_path = 'pp_mobileseg_base_camvid_1024x1024_model.onnx'
    save_path = 'pred.png'
    infer_onnx(img_path, onnx_path, save_path)