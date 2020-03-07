#!/usr/bin/python3

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
sys.path.append(os.getcwd() + '/../src')
from config import cfg
from prior_box import PriorBox
from nms import nms
from utils import decode
from yufacedetectnet import YuFaceDetectNet

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def load_image(file_id):
    if type(file_id)==str:
        img_raw = cv2.imread(file_id, cv2.IMREAD_COLOR)
    elif type(file_id)==np.ndarray:
        img_raw = file_id.copy()
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape

    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    scale = torch.Tensor([im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height ])
    
    scale = scale.to(device)
    return img, scale, img_raw, im_height, im_width
    
def get_prediction(img, scale, im_height, im_width, print_messages=False):
    loc, conf = net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > 0.3)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    scores = scores[order]
    if print_messages:
        print('there are', len(boxes), 'candidates')
    return boxes, scores
    
def do_NMS(boxes, scores, print_messages=False):
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    selected_idx = np.array([0,1,2,3,14])
    keep = nms(dets[:,selected_idx], 0.3)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:750, :]

    # save dets
    face_cc = 0;
    for k in range(dets.shape[0]):
        if dets[k, 14] < 0.8:
            continue
        xmin = dets[k, 0]
        ymin = dets[k, 1]
        xmax = dets[k, 2]
        ymax = dets[k, 3]
        score = dets[k, 14]
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        face_cc =  face_cc + 1
        if print_messages:
            print('{}: {:.3f} {:.3f} {:.3f} {:.3f} {:.10f}'.format(face_cc, xmin, ymin, w, h, score))
    return dets

def draw_image(dets, img_raw, file_id, save_result=True):
    # draw image
    for b in dets:
        if b[14] < 0.8:
            continue
        text = "{:.4f}".format(b[14])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        
        cv2.circle(img_raw, (b[4], b[4 + 1]), 2, (255, 0, 0), 2)
        cv2.circle(img_raw, (b[4 + 2], b[4 + 3]), 2, (0, 0, 255), 2)
        cv2.circle(img_raw, (b[4 + 4], b[4 + 5]), 2, (0, 255, 255), 2)
        cv2.circle(img_raw, (b[4 + 6], b[4 + 7]), 2, (255, 255, 0), 2)
        cv2.circle(img_raw, (b[4 + 8], b[4 + 9]), 2, (0, 255, 0), 2)
        
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    if save_result:
        cv2.imwrite('./output_image/{}'.format(file_id.split('/')[-1]), img_raw)
    return img_raw

def deal_img(file_id, save_result=True, print_messages=False):
    # pipeline
    img, scale, img_raw, im_height, im_width = load_image(file_id)
    boxes, scores = get_prediction(img, scale, im_height, im_width, print_messages)
    dets = do_NMS(boxes, scores, print_messages)
    result = draw_image(dets, img_raw, file_id, save_result)
    return result

def deal_video(filepath=0, output_path='./output_video/', show_info=True):
    try:
        video_reader = cv2.VideoCapture(filepath)
    except:
        raise FileNotFoundError (filepath)
    fps, hei, wid = int(video_reader.get(5)), int(video_reader.get(4)), int(video_reader.get(3))
    if filepath == 0:
        print('Getting camera input...')
        video_writer = cv2.VideoWriter(output_path + 'camera.mp4', cv2.VideoWriter_fourcc(*'XVID'), 15, (wid, hei))
    else:
        filename = filepath.split('/')[-1]
        video_writer = cv2.VideoWriter(output_path + filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (wid, hei))
    while True:
        ret, frame = video_reader.read()
        if ret != True:
            break
        tik = time.time() #####################################################
        detections = deal_img(frame, save_result=False, print_messages=False)
        tok = time.time() #####################################################
        info = 'FPS: {:.1f}, takes: {:.1f}ms.'.format( 1./(tok-tik), 1000*(tok-tik))
        if show_info:
            cv2.putText(detections, text=info, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= 0.9, color=(0, 0, 255), thickness=2)
        if not filepath:
            cv2.imshow("Press 'Esc' to exit.", detections)
        video_writer.write(detections)
        if cv2.waitKey(1) == 27:
            break
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()
    return None
    
if __name__ == '__main__':
    device = torch.device('cuda:0') 
    torch.set_grad_enabled(False)
    # net and model, initialize detector
    net = YuFaceDetectNet(phase='test', size=None)
    net = load_model(net, 'weights/yunet_final.pth', load_to_cpu=True)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    net = net.to(device)
    
    file_list = os.listdir('./input_image/')
    for i in file_list:
        print('Dealing with {}...'.format(i))
        file_id = './input_image/'+ i
        result = deal_img(file_id, save_result=True, print_messages=False)
    
    file_list = os.listdir('./input_video/')
    for i in file_list:
        print('Dealing with {}...'.format(i))
        file_id = './input_video/'+ i
        #deal_video(file_id, show_info=True)
    
    #deal_video(0, show_info=True)
    
    
    
    