import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "rbbox_overlaps.hpp":
    void _overlaps(np.float32_t*, np.float32_t*, np.float32_t*, int, int, int)
    void _overlaps_3d(np.float32_t*, np.float32_t*, np.float32_t*, int, int, int)
    void _rotate_nms_3d(np.int32_t*, int*, np.float32_t*, int, int, float, int)

def py_rbbox_overlaps (np.ndarray[np.float32_t, ndim=2] boxes, np.ndarray[np.float32_t, ndim=2] query_boxes, np.int32_t device_id=0):
    cdef int N = boxes.shape[0]
    cdef int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] overlaps = np.zeros((N, K), dtype = np.float32)
    _overlaps(&overlaps[0, 0], &boxes[0, 0], &query_boxes[0, 0], N, K, device_id)
    return overlaps

def py_rbbox_overlaps_3d (np.ndarray[np.float32_t, ndim=2] boxes, np.ndarray[np.float32_t, ndim=2] query_boxes, np.int32_t device_id=0):
    cdef int N = boxes.shape[0]
    cdef int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] overlaps = np.zeros((N, K), dtype = np.float32)
    _overlaps_3d(&overlaps[0, 0], &boxes[0, 0], &query_boxes[0, 0], N, K, device_id)
    return overlaps

def py_rotate_nms_3d(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh, np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1] # 8 = x y z l h w r score
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] \
        keep = np.zeros(boxes_num, dtype=np.int32)
    # sort by score
    cdef np.ndarray[np.float32_t, ndim=1] \
        scores = dets[:, 5]
    cdef np.ndarray[np.int_t, ndim=1] \
        order = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets = dets[order, :]
    # cal
    _rotate_nms_3d(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)
    keep = keep[:num_out]
    return list(order[keep])
