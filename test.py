import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import cv2
import os
import json
from torch.optim import *
import numpy as np
from sklearn import metrics
import os
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import xml.etree.ElementTree as ET
import av
# import torchaudio
import numpy as np
from fractions import Fraction


class EZVSL(nn.Module):
    def __init__(self, tau, dim):
        super(EZVSL, self).__init__()
        self.tau = tau

        # Vision model
        self.imgnet = resnet18(pretrained=True)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_proj = nn.Conv2d(512, dim, kernel_size=(1, 1))

        # Audio model
        self.audnet = resnet18()
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.audnet.fc = nn.Identity()
        self.aud_proj = nn.Linear(512, dim)

        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_proj, self.aud_proj]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def max_xmil_loss(self, img, aud):
        B = img.shape[0]
        Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau
        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
        labels = torch.arange(B).long().to(img.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)
        return loss, Slogits

    def forward(self, image, audio):
        # Image
        img = self.imgnet(image).unflatten(1, (512, 7, 7))
        img = self.img_proj(img)
        img = nn.functional.normalize(img, dim=1)

        # Audio
        aud = self.audnet(audio)
        aud = self.aud_proj(aud)
        aud = nn.functional.normalize(aud, dim=1)

        # Compute loss
        loss, logits = self.max_xmil_loss(img, aud)

        # Compute avl maps
        with torch.no_grad():
            B = img.shape[0]
            Savl = logits[torch.arange(B), torch.arange(B)]

        return loss, Savl




class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224))
        infer_map[infer >= thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap==0)))

        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))

    def finalize_AUC(self):
        cious = [np.sum(np.array(self.ciou) >= 0.05*i) / len(self.ciou)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def finalize_AP50(self):
        ap50 = np.mean(np.array(self.ciou) >= 0.5)
        return ap50

    def finalize_cIoU(self):
        ciou = np.mean(np.array(self.ciou))
        return ciou

    def clear(self):
        self.ciou = []


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


# def visualize(raw_image, boxes):
#     import cv2
#     boxes_img = np.uint8(raw_image.copy())[:, :, ::-1]
#
#     for box in boxes:
#
#         xmin,ymin,xmax,ymax = int(box[0]),int(box[1]),int(box[2]),int(box[3])
#
#         cv2.rectangle(boxes_img[:, :, ::-1], (xmin, ymin), (xmax, ymax), (0,0,255), 1)
#
#     return boxes_img[:, :, ::-1]


def build_optimizer_and_scheduler_adam(model, args):
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(optimizer_grouped_parameters, lr=args.init_lr)
    scheduler = None
    return optimizer, scheduler


def build_optimizer_and_scheduler_sgd(model, args):
    optimizer_grouped_parameters = model.parameters()
    optimizer = SGD(optimizer_grouped_parameters, lr=args.init_lr)
    scheduler = None
    return optimizer, scheduler


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode='w', encoding='utf-8') as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def save_iou(iou_list, suffix, output_dir):
    # sorted iou
    sorted_iou = np.sort(iou_list).tolist()
    sorted_iou_indices = np.argsort(iou_list).tolist()
    file_iou = open(os.path.join(output_dir,"iou_test_{}.txt".format(suffix)),"w")
    for indice, value in zip(sorted_iou_indices, sorted_iou):
        line = str(indice) + ',' + str(value) + '\n'
        file_iou.write(line)
    file_iou.close()


def open_audio_av(path):
    container = av.open(path)
    for stream in container.streams.video:
        stream.codec_context.thread_type = av.codec.context.ThreadType.NONE
        stream.codec_context.thread_count = 1
    for stream in container.streams.audio:
        stream.codec_context.thread_type = av.codec.context.ThreadType.NONE
        stream.codec_context.thread_count = 1
    return container


def load_audio_av(path=None, container=None, rate=None, start_time=None, duration=None, layout="mono"):
    if container is None:
        container = av.open(path)
    audio_stream = container.streams.audio[0]

    # Parse metadata
    _ss = audio_stream.start_time * audio_stream.time_base if audio_stream.start_time is not None else 0.
    _dur = audio_stream.duration * audio_stream.time_base
    _ff = _ss + _dur
    _rate = audio_stream.rate

    if rate is None:
        rate = _rate
    if start_time is None:
        start_time = _ss
    if duration is None:
        duration = _ff - start_time
    duration = min(duration, _ff - start_time)
    end_time = start_time + duration

    resampler = av.audio.resampler.AudioResampler(format="s16p", layout=layout, rate=rate)

    # Read data
    chunks = []
    container.seek(int(start_time * av.time_base))
    for frame in container.decode(audio=0):
        chunk_start_time = frame.pts * frame.time_base
        chunk_end_time = chunk_start_time + Fraction(frame.samples, frame.rate)
        if chunk_end_time < start_time:   # Skip until start time
            continue
        if chunk_start_time > end_time:       # Exit if clip has been extracted
            break

        try:
            frame.pts = None
            if resampler is not None:
                chunks.append((chunk_start_time, resampler.resample(frame).to_ndarray()))
            else:
                chunks.append((chunk_start_time, frame.to_ndarray()))
        except AttributeError:
            break

    # Trim for frame accuracy
    audio = np.concatenate([af[1] for af in chunks], 1)
    ss = int((start_time - chunks[0][0]) * rate)
    t = int(duration * rate)
    if ss < 0:
        audio = np.pad(audio, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
        ss = 0
    audio = audio[:, ss: ss+t]

    # Normalize to [-1, 1]
    audio = audio / np.iinfo(audio.dtype).max

    return audio, rate


def audio_info_av(inpt, audio=None, format=None):
    container = inpt
    if isinstance(inpt, str):
        try:
            container = av.open(inpt, format=format)
        except av.AVError:
            return None, None

    audio_stream = container.streams.audio[audio]
    time_base = audio_stream.time_base
    duration = audio_stream.duration * time_base
    start_time = audio_stream.start_time * time_base
    channels = audio_stream.channels
    fps = audio_stream.rate
    chunk_size = audio_stream.frame_size
    chunks = audio_stream.frames
    meta = {'channels': channels,
            'fps': fps,
            'start_time': start_time,
            'duration': duration,
            'chunks': chunks,
            'chunk_size': chunk_size}
    return meta

def load_image(path):
    return Image.open(path).convert('RGB')


def load_spectrogram(path, dur=3.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram


class AudioVisualDataset(Dataset):
    def __init__(self, image_files, audio_files, image_path, audio_path, audio_dur=3., image_transform=None, audio_transform=None):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path
        self.audio_dur = audio_dur

        self.audio_files = audio_files
        self.image_files = image_files


        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def getitem(self, idx):
        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = os.path.join(self.image_path, self.image_files[idx])
        frame = self.image_transform(load_image(img_fn))

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
        spectrogram = self.audio_transform(load_spectrogram(audio_fn))

        #bboxes = {}
        #if self.all_bboxes is not None:
            #bboxes['bboxes'] = self.all_bboxes[file_id]
            #bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file_id], self.bbox_format)

        return frame, spectrogram, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


def get_train_dataset(args):
    audio_path = f"{args.train_data_path}/audio/"
    image_path = f"{args.train_data_path}/frames/"

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path) if fn.endswith('.jpg')}
    avail_files = audio_files.intersection(image_files)
    print(f"{len(avail_files)} available files")

    # Subsample if specified
    if args.trainset.lower() in {'vggss', 'flickr'}:
        pass    # use full dataset
    else:
        subset = set(open(f"metadata/{args.trainset}.txt").read().splitlines())
        avail_files = avail_files.intersection(subset)
        print(f"{len(avail_files)} valid subset files")
    avail_files = sorted(list(avail_files))
    audio_files = sorted([dt+'.wav' for dt in avail_files])
    image_files = sorted([dt+'.jpg' for dt in avail_files])

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform
    )


def get_test_dataset(args):
    audio_path = args.test_data_path + 'audio/'
    image_path = args.test_data_path + 'frames/'


    #  Retrieve list of audio and video files
    testset = {''}
    print(type(testset))

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    print((audio_files))
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
    print((image_files))
    avail_files = audio_files.intersection(image_files)
    print(len(avail_files))

    testset = sorted(list(avail_files))
    image_files = [dt+'.jpg' for dt in testset]
    audio_files = [dt+'.wav' for dt in testset]
    print(len(testset))

  
    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        #all_bboxes=all_bboxes,
        #bbox_format=bbox_format
    )


def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
import cv2


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='ezvsl_vggss', help='experiment name (experiment folder set to "args.model_dir/args.experiment_name)"')
    parser.add_argument('--save_visualizations', action='store_true', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')

    # Dataset
    parser.add_argument('--testset', default='flickr', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of data')
    #parser.add_argument('--test_gt_path', default='', type=str)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')

    # Model
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    # viz_dir = os.path.join(model_dir, 'vizzz')
    #
   
    viz_dir = '/Users/christos/EZDRUMMER/vizzone/'

    # Models
    audio_visual_model = EZVSL(args.tau, args.out_dim)

    from torchvision.models import resnet18
    object_saliency_model = resnet18(pretrained=True)
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            audio_visual_model.cuda(args.gpu)
            object_saliency_model.cuda(args.gpu)
            audio_visual_model = torch.nn.parallel.DistributedDataParallel(audio_visual_model, device_ids=[args.gpu])
            object_saliency_model = torch.nn.parallel.DistributedDataParallel(object_saliency_model, device_ids=[args.gpu])

    # Load weights
    ckp_fn = os.path.join(model_dir, 'best.pth')
    if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu')
        audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        print(f'loaded from {os.path.join(model_dir, "best.pth")}')
    else:
        print(f"Checkpoint not found: {ckp_fn}")

    # Dataloader
    #testdataset - get_test_dataset(args)
    testdataset = get_test_dataset(args)
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(len(testdataloader))
    print("Loaded dataloader.")

    validate(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args)


@torch.no_grad()
def validate(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args):
    audio_visual_model.train(False)
    object_saliency_model.train(False)

    evaluator_av = utils.Evaluator()
    evaluator_obj = utils.Evaluator()
    evaluator_av_obj = utils.Evaluator()
    for step, (image, spec,  name) in enumerate(testdataloader):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        # Compute S_AVL
        heatmap_av = audio_visual_model(image.float(), spec.float())[1].unsqueeze(1)
        heatmap_av = F.interpolate(heatmap_av, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_av = heatmap_av.data.cpu().numpy()

        # Compute S_OBJ
        img_feat = object_saliency_model(image)
        heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_obj = heatmap_obj.data.cpu().numpy()

        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_av = utils.normalize_img(heatmap_av[i, 0]) #audio-visual similarity
            pred_obj = utils.normalize_img(heatmap_obj[i, 0]) # object localization map
            pred_av_obj = utils.normalize_img(pred_av * args.alpha + pred_obj * (1 - args.alpha)) # Sezvsl

            #gt_map = bboxes['gt_map'].data.cpu().numpy()

            thr_av = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] * 0.5)]
            #evaluator_av.cal_CIOU(pred_av, gt_map, thr_av)
            evaluator_av.cal_CIOU(pred_av, thr_av)

            thr_obj = np.sort(pred_obj.flatten())[int(pred_obj.shape[0] * pred_obj.shape[1] * 0.5)]
            evaluator_obj.cal_CIOU(pred_obj, thr_obj)

            thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * 0.5)]
            evaluator_av_obj.cal_CIOU(pred_av_obj, thr_av_obj)

            if args.save_visualizations:
                denorm_image = inverse_normalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                denorm_image = (denorm_image*255).astype(np.uint8)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_image.jpg'), denorm_image)

                

                # visualize heatmaps
                heatmap_img = np.uint8(pred_av*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                print(fin,"fin is ")
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av.jpg'), fin)

                heatmap_img = np.uint8(pred_obj*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_obj.jpg'), fin)

                heatmap_img = np.uint8(pred_av_obj*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av_obj.jpg'), fin)

        print(f'{step+1}/{len(testdataloader)}: map_av={evaluator_av.finalize_AP50():.2f} map_obj={evaluator_obj.finalize_AP50():.2f} map_av_obj={evaluator_av_obj.finalize_AP50():.2f}')

    def compute_stats(eval):
        mAP = eval.finalize_AP50()
        ciou = eval.finalize_cIoU()
        auc = eval.finalize_AUC()
        return mAP, ciou, auc

    print('AV: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av)))
    print('Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_obj)))
    print('AV_Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av_obj)))

    utils.save_iou(evaluator_av.ciou, 'av', viz_dir)
    utils.save_iou(evaluator_obj.ciou, 'obj', viz_dir)
    utils.save_iou(evaluator_av_obj.ciou, 'av_obj', viz_dir)


class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


if __name__ == "__main__":
    main(get_arguments())


