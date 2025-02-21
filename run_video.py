import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    margin_width = 50

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        
        filename = os.path.basename(filename)
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

        # keep track of numpy arrays representing depth
        depth_arrays = []
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            # here

            # depth = depth_anything.infer_image(raw_frame, args.input_size)

            # depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]

            with torch.no_grad():
                depth = depth_anything(raw_frame)

            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]

            depth_shape = depth.shape

            depth = depth.cpu().numpy()  # Move to CPU and convert to NumPy
            depth_arrays.append(np.reshape(depth, (1, depth_shape[0], depth_shape[1])))

            # depth_arrays.append(np.reshape(depth, (1, depth_shape[0], depth_shape[1])))
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            if args.pred_only:
                out.write(depth)
            else:
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth])
                
                out.write(combined_frame)
            
            # here
        
        raw_video.release()
        out.release()

        # save depth as tensor
        npz_depth_file = np.concatenate(depth_arrays)

        depth_tensor_output = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".npz")
        np.savez(depth_tensor_output, depth=npz_depth_file)
