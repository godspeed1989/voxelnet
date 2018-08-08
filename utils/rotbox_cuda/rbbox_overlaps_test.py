import numpy as np
import cv2
from rbbox_overlaps import py_rbbox_overlaps
from rbbox_overlaps import py_rbbox_overlaps_3d
from rbbox_overlaps import py_rotate_nms_3d

def rbbx_overlaps(boxes, query_boxes):
    '''
    Parameters
    ----------------
    boxes: (N, 5) --- x_ctr, y_ctr, height, width, angle
    query: (K, 5) --- x_ctr, y_ctr, height, width, angle
    ----------------
    Returns
    ----------------
    Overlaps (N, K) IoU
    '''
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype = np.float32)

    for k in range(K):
        query_area = query_boxes[k, 2] * query_boxes[k, 3]
        for n in range(N):
            box_area = boxes[n, 2] * boxes[n, 3]
            #IoU of rotated rectangle
            #loading data anti to clock-wise
            rn = ((boxes[n, 0], boxes[n, 1]), (boxes[n, 3], boxes[n, 2]), -boxes[n, 4])
            rk = ((query_boxes[k, 0], query_boxes[k, 1]), (query_boxes[k, 3], query_boxes[k, 2]), -query_boxes[k, 4])
            int_pts = cv2.rotatedRectangleIntersection(rk, rn)[1]

            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints = True)
                int_area = cv2.contourArea(order_pts)
                overlaps[n, k] = int_area * 1.0 / (query_area + box_area - int_area)
    return overlaps

def cal_z_intersect(cz1, h1, cz2, h2):
    b1z1, b1z2 = cz1 - h1 / 2, cz1 + h1 / 2
    b2z1, b2z2 = cz2 - h2 / 2, cz2 + h2 / 2
    if b1z1 > b2z2 or b2z1 > b1z2:
        return 0
    elif b2z1 <= b1z1 <= b2z2:
        if b1z2 <= b2z2:
            return h1
        else:
            return b2z2 - b1z1
    elif b1z1 < b2z1 < b1z2:
        if b2z2 <= b1z2:
            return h2
        else:
            return b1z2 - b2z1
    return 0

def cal_box_share(box, query_box):
    rn = ((box[0], box[1]), (box[3], box[2]), -box[4])
    rk = ((query_box[0], query_box[1]), (query_box[3], query_box[2]), -query_box[4])
    int_pts = cv2.rotatedRectangleIntersection(rk, rn)[1]

    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints = True)
        int_area = cv2.contourArea(order_pts)
        return int_area
    return 0

def cal_rbbox3d_iou(boxes3d, gt_boxes3d):
    # Inputs:
    #   boxes3d: (N1, 7) x,y,z,h,w,l,r
    #   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    N1 = len(boxes3d)
    N2 = len(gt_boxes3d)
    output = np.zeros((N1, N2), dtype=np.float32)
    for idx in range(N1):
        for idy in range(N2):
            area1 = boxes3d[idx, 4] * boxes3d[idx, 5]
            area2 = gt_boxes3d[idy, 4] * gt_boxes3d[idy, 5]
            share = cal_box_share(boxes3d[idx, [0, 1, 4, 5, 6]], gt_boxes3d[idy, [0, 1, 4, 5, 6]])
            z1, h1, z2, h2 = boxes3d[idx, 2], boxes3d[idx, 3], gt_boxes3d[idy, 2], gt_boxes3d[idy, 3]
            z_intersect = cal_z_intersect(z1, h1, z2, h2)
            output[idx, idy] = share * z_intersect / (area1 * h1 + area2 * h2 - share * z_intersect)
    return output

def cal_nms_3d(dets, threshold):
    '''
    Parameters
    ----------------
    dets: (N, 8) x y z l h w r score
    threshold: 0.7 or 0.5 IoU
    ----------------
    Returns
    ----------------
    keep: keep the remaining index of dets
    '''
    keep = []
    scores = dets[:, -1]
    # 按照score置信度降序排序
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype = np.int)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        suppressed[i] = 1
        #r1 = ((dets[i,0],dets[i,1]),(dets[i,3],dets[i,2]),dets[i,4])
        #area_r1 = dets[i,2]*dets[i,3]
        box1 = dets[i, :7]
        for _j in range(_i+1, ndets):
            #tic = time.time()
            j = order[_j]
            if suppressed[j] == 1:
                continue
            box2 = dets[j, :7]

            area1 = box1[4] * box1[5]
            area2 = box2[4] * box2[5]
            share = cal_box_share(box1[[0, 1, 4, 5, 6]], box2[[0, 1, 4, 5, 6]])
            z1, h1, z2, h2 = box1[2], box1[3], box2[2], box2[3]
            z_intersect = cal_z_intersect(z1, h1, z2, h2)
            ovr = share * z_intersect / (area1 * h1 + area2 * h2 - share * z_intersect)

            if ovr > threshold:
                suppressed[j] = 1
    return keep

if __name__ == "__main__":

    # x, y, h, w  x->w  y->h
    boxes = np.array([
            [60.0, 60.0, 120.0,  120.0, 0.0], # 4 pts
            [60.0, 180.0, 125.0,  120.0, 0.0], # 4 pts
            [60.0, 180.0, 120.0,  125.0, 0.0], # 4 pts
            [60.0, 60.0, 80.0,   80.0, 0.0], # 4 pts
            [60.0, 60.0, 100.0,  100.0, 10.0], # 4 pts
            [50.0, 50.0, 100.0, 100.0, 45.0], # 8 pts
            [80.0, 50.0, 100.0, 100.0, 0.0], # overlap 4 edges
            [50.0, 50.0, 200.0, 50.0, 45.0], # 6 edges
            [200.0, 200.0, 100.0, 100.0, 0], # no intersection
            [60.0, 60.0, 100.0,  100.0, 0.0], # 4 pts
            [50.0, 50.0, 100.0, 100.0, 45.0], # 8 pts
            ], dtype = np.float32)
    '''
    o1 = py_rbbox_overlaps(np.ascontiguousarray(boxes, dtype=np.float32),
                           np.ascontiguousarray(boxes, dtype=np.float32))
    o2 = rbbx_overlaps(boxes, boxes)
    print('1', np.allclose(o1, o2))
    '''
    # x y z h w l r   z->h  x->l  y->w
    # 0 1     4 5 6
    # x y     h w
    boxes_3d = np.array([
            [60.0, 60.0, 0, 1, 100.0,  100.0, 0.0], # 4 pts
            [60.0, 80.0, 0, 1, 100.0,  100.0, 0.0], # 4 pts
            [60.0, 60.0, 0, 1, 80.0,   80.0, 0.0], # 4 pts
            [50.0, 50.0, 0, 1, 200.0,  50.0, 45.0], # 6 edges
            [200.0, 200.0, 0, 1, 100.0,  100.0, 0], # no intersection
            [60.0, 60.0, 0, 1, 100.0,  100.0, 0.0], # 4 pts
            [50.0, 50.0, 0, 1, 100.0,  100.0, 45.0], # 8 pts
            ], dtype = np.float32)
    '''
    for i in range(10000):
        cnt = boxes_3d.shape[0]
        a, b = np.random.randint(cnt), np.random.randint(cnt)
        s = a if a < b else b
        e = a if a > b else b
        e = e + 1
        o3 = cal_rbbox3d_iou(boxes_3d[s:e], boxes_3d)
        o4 = py_rbbox_overlaps_3d(np.ascontiguousarray(boxes_3d[s:e], dtype=np.float32),
                                  np.ascontiguousarray(boxes_3d, dtype=np.float32))
        assert np.allclose(o3, o4)
    print('2', True)
    '''
    aa, bb  = 0, 0
    for i in range(10000):
        scores = np.random.rand(len(boxes_3d))
        scores = np.expand_dims(scores, axis=-1)
        boxes_3d_score = np.hstack((boxes_3d, scores))

        cnt = boxes_3d_score.shape[0]
        assert 8 == boxes_3d_score.shape[1]
        a, b = np.random.randint(cnt), np.random.randint(cnt)
        s = a if a < b else b
        e = a if a > b else b
        e = e + 1
        dets = boxes_3d_score[s:e]
        nms_thresh = np.random.rand()
        keep1 = py_rotate_nms_3d(np.ascontiguousarray(dets, dtype=np.float32), nms_thresh)
        keep2 = cal_nms_3d(dets, nms_thresh)
        if len(keep1) > len(keep2):
            print(len(dets), keep1, '>', keep2)
            aa = aa + 1
        if len(keep1) < len(keep2):
            print(len(dets), keep1, '<', keep2)
            bb = bb + 1
    print('3', True)
    print(aa, '++', bb)
