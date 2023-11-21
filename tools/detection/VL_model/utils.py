import torch
import numpy as np
from tqdm import tqdm
import clip
from PIL import Image
import copy


COCO_SPLIT = dict(
    ALL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    NOVEL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
                   'cow', 'bottle', 'chair', 'couch', 'potted plant',
                   'dining table', 'tv'),
    BASE_CLASSES=('truck', 'traffic light', 'fire hydrant', 'stop sign',
                  'parking meter', 'bench', 'elephant', 'bear', 'zebra',
                  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
                  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                  'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush'))

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',

    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]


def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'

def scale_box(box, scale, max_H=np.inf, max_W=np.inf):
    # box: x0, y0, x1, y1
    # scale: float
    print(len(box))
    x0, y0, x1, y1, score = box

    cx = ((x0 + x1) / 2)
    cy = ((y0 + y1) / 2)
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale

    new_x0 = max(cx - bw / 2, 0)
    new_y0 = max(cy - bh / 2, 0)
    new_x1 = min(cx + bw / 2, max_W)
    new_y1 = min(cy + bh / 2, max_H)

    return [new_x0, new_y0, new_x1, new_y1]

#### get novel classes id
def get_coco_ids_by_order(coco_complete, cat_name_list):
    catIDs_list = list()
    for catName in cat_name_list:
        catID = coco_complete.coco.getCatIds(catNms=[catName])
        if len(catID) != 1:
            print('%s is not valid cat name' % catName)
        else:
            catIDs_list.append(catID[0])

    return catIDs_list


def get_novel_labels_by_catIDs(coco_complete, catIds):

    novel_labels = []
    cat2label = coco_complete.cat2label
    for id in catIds.tolist():
        novel_labels.append(cat2label[id])
    return novel_labels


def build_text_embedding(model, categories, templates, add_this_is=False, show_process=True):   # device='cpu'
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        if show_process:
            print('Building text embeddings...')
        for catName in (tqdm(categories) if show_process else categories):
            texts = [template.format(catName, article=article(catName)) for template in templates]
            if add_this_is:
                texts = ['This is ' + text if text.startswith('a') or text.startswith('the') else text
                         for text in texts]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
            # if device != 'cpu':
            #     #idx = int(device.split(":", 1)[-1])
            #     #texts = texts.cuda(idx)
            #     texts = texts.cuda()

            text_embeddings = model.encode_text(texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)

        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()
        # if device != 'cpu':
        #     #idx = int(device.split(":", 1)[-1])
        #     #all_text_embeddings = all_text_embeddings.cuda(idx)
        #     all_text_embeddings = all_text_embeddings.cuda()

        return all_text_embeddings.transpose(dim0=0, dim1=1)


def CLIP_score_multi_scale(clip_model, img_patch_scalelist, text_features, softmax_t=0.01):
    # img_patch_scalelist: [ [n*patches for scale 1], [n*patches for scale 2], ...]
    # patchNum = img_patch_scalelist[0].shape[0]

    patchNum = len(img_patch_scalelist[0])
    patch_per_split = 1000

    splitIdxList = [i*patch_per_split for i in range(1 + patchNum//patch_per_split)]
    if splitIdxList[-1] != patchNum:
        splitIdxList.append(patchNum)

    allSimilarity = []

    for sp_idx in range(len(splitIdxList) - 1):
        startIdx = splitIdxList[sp_idx]
        endIdx = splitIdxList[sp_idx + 1]

        image_features = None
        for s_id, imgPatchesList in enumerate(img_patch_scalelist):
            imgPatches = torch.cat(imgPatchesList[startIdx:endIdx], dim=0)

            with torch.no_grad():
                curImgFeat = clip_model.module.encode_image(imgPatches)
                curImgFeat /= curImgFeat.norm(dim=-1, keepdim=True)

            if image_features is None:
                image_features = curImgFeat.clone()
            else:
                image_features += curImgFeat     # Fusing multi-scale image features

        sacleNum = len(img_patch_scalelist)
        image_features /= sacleNum
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = ((1 / softmax_t) * image_features @ text_features.T).softmax(dim=-1)
        allSimilarity.append(similarity)

    allSimilarity = torch.cat(allSimilarity, dim=0)
    return allSimilarity


def get_CLIP_pred_for_det_results(input_imgs, det_bboxes, det_labels, CLIP_model, preprocess, clip_text_embed, usedCatIds_inOrder,
                                  cat2label, box_scalelist=[1, 1.5], topK_clip_scores=1, device='cuda'):
    '''
    input_img: from cv2.imread, in BGR
    proposal_boxes: [[xyxy], [xyxy], ...]
    pp_scores: objectness scores for each region proposal
    '''
    verBoxList = []
    verLabelIdList = []

    for i, (input_img, det_bbox) in enumerate(zip(input_imgs, det_bboxes)):
        if len(det_bbox):
            input_img = input_img.cpu().numpy()
            height, width = input_img.shape[:2]
            pilImg = Image.fromarray(input_img[:, :, ::-1])  # RGB

            usedCatNum = len(usedCatIds_inOrder)

            curBoxList = list()
            curCLIPBboxList = list()

            clipInput_list_scalelist = [[] for i in range(len(box_scalelist))]
            #det_boxes
            for b_idx, box in enumerate(det_bbox.tolist()):
                scaledBox = scale_box(box, 1, max_H=height, max_W=width)  # ensure every box is in the image
                if scaledBox[2] - scaledBox[0] >= 5 and scaledBox[3] - scaledBox[1] >= 5:
                    curBoxList.append(scaledBox)
                    # add scales
                    for scale_id, boxScale in enumerate(box_scalelist):
                        scaledBox = scale_box(box, boxScale, max_H=height, max_W=width)
                        cropImg = pilImg.crop(scaledBox)

                        clipInput = preprocess(cropImg).unsqueeze(0).to(device)
                        clipInput_list_scalelist[scale_id].append(clipInput)

            if len(curBoxList) > 0:
                allSimilarity = CLIP_score_multi_scale(CLIP_model, clipInput_list_scalelist, clip_text_embed)
                clipPreLabel = copy.deepcopy(det_labels[i])

                ############### merge CLIP score and det bboxes
                for b_idx, box in enumerate(curBoxList):
                    clipScores, indices = allSimilarity[b_idx][:usedCatNum].topk(topK_clip_scores)
                    box = torch.cat((torch.tensor(box).to(device), clipScores), 0)
                    curCLIPBboxList.append(box)
                    clipPreLabel[b_idx] = cat2label[usedCatIds_inOrder[indices]]
                verBoxList.append(torch.stack(curCLIPBboxList))
                verLabelIdList.append(clipPreLabel)
        else:
            verBoxList.append(det_bbox)
            verLabelIdList.append(det_labels[i])

    return verBoxList, verLabelIdList


# def get_CLIP_pred_for_det_results(input_img, det_boxes, CLIP_model, preprocess, clip_text_embed, usedCatIds_inOrder,
#                                 box_scalelist=[1, 1.5], topK_clip_scores=1, device='cuda'):
#     '''
#     input_img: from cv2.imread, in BGR
#     proposal_boxes: [[xyxy], [xyxy], ...]
#     pp_scores: objectness scores for each region proposal
#     '''
#
#     input_img = input_img.cpu().numpy()
#     height, width = input_img.shape[:2]
#     pilImg = Image.fromarray(input_img[:, :, ::-1])  # RGB
#
#     usedCatNum = len(usedCatIds_inOrder)
#
#     curBoxList = list()
#     curRPNScoreList = list()
#     curCLIPScoreList = list()
#     curPredCatIdList = list()
#
#     clipInput_list_scalelist = [[] for i in range(len(box_scalelist))]
#     #det_boxes
#     for b_idx, box in enumerate(det_boxes.tolist()):
#         scaledBox = scale_box(box, 1, max_H=height, max_W=width)  # ensure every box is in the image
#         if scaledBox[2] - scaledBox[0] >= 5 and scaledBox[3] - scaledBox[1] >= 5:
#             curBoxList.append(scaledBox)
#             curRPNScoreList.append(box[4])
#
#             # add scales
#             for scale_id, boxScale in enumerate(box_scalelist):
#                 scaledBox = scale_box(box, boxScale, max_H=height, max_W=width)
#                 cropImg = pilImg.crop(scaledBox)
#
#                 clipInput = preprocess(cropImg).unsqueeze(0).to(device)
#                 clipInput_list_scalelist[scale_id].append(clipInput)
#
#     if len(curBoxList) > 0:
#         allSimilarity = CLIP_score_multi_scale(CLIP_model, clipInput_list_scalelist, clip_text_embed)
#
#         ############### merge CLIP and RPN scores
#         for b_idx, box in enumerate(curBoxList):
#             clipScores, indices = allSimilarity[b_idx][:usedCatNum].topk(topK_clip_scores)
#
#             curCLIPScoreList.append(clipScores.cpu().numpy().tolist())
#             curPredCatIdList.append(usedCatIds_inOrder[indices.cpu().numpy()].tolist())
#
#     return curBoxList, curRPNScoreList, curCLIPScoreList, curPredCatIdList
