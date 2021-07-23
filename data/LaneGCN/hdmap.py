import numpy as np
import copy
from scipy import sparse
from argoverse.map_representation.map_api import ArgoverseMap

class GraphExtractor(object):
    def __init__(self, config, mode='train'):
        self.am = ArgoverseMap()
        self.config = config
        self.mode = mode

    def __del__(self):
        del self.am

    def dilated_nbrs(self, nbr, num_nodes, num_scales):
        data = np.ones(len(nbr['u']), np.bool)
        csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

        mat = csr
        nbrs = []
        for i in range(1, num_scales):
            mat = mat * mat

            nbr = dict()
            coo = mat.tocoo()
            nbr['u'] = coo.row.astype(np.int16)
            nbr['v'] = coo.col.astype(np.int16)
            nbrs.append(nbr)
        return nbrs

    def extract(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)

        """Get all lane within self.config['pred_range'], convert centerline and polygon to rotated and biased"""
        # what's polygon
        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        """Lane feature: ctrs(position), feats(shape), turn, control, intersect"""
        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

        # -------------------------node_idcs---------------------
        node_idcs = []
        count = 0
        for ctr in ctrs:  # node_idcs: list, i-th element: i-th lane nodes ids
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count
        # -------------------------lane_idcs---------------------
        # lane[idc] = a means idc-th node belongs to the a-th lane
        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))  # TODO: what does lane_idcs do?
        lane_idcs = np.concatenate(lane_idcs, 0)

        # **********************************Map Related work***************************
        # =========================================
        # ==============Hdmap Graph Build==========
        # =========================================

        # -------all in all, pairs is for lanes; no pairs is for lanes--------
        # ---------------------------pre and suc for lanes--------------------
        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int16)
        suc_pairs = np.asarray(suc_pairs, np.int16)
        left_pairs = np.asarray(left_pairs, np.int16)
        right_pairs = np.asarray(right_pairs, np.int16)

        # ---------------------------pre and suc for nodes--------------------
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]

            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1]) # v is the pre of u, v is src, u is dest

            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        pre['u'] = np.asarray(pre['u'], dtype=np.int16)
        pre['v'] = np.asarray(pre['v'], dtype=np.int16)
        suc['u'] = np.asarray(suc['u'], dtype=np.int16)
        suc['v'] = np.asarray(suc['v'], dtype=np.int16)

        # -------------------dilate pre and suc: opition 1--------------------
        dilated_pre = [pre]
        dilated_pre += self.dilated_nbrs(pre, num_nodes, self.config['num_scales'])
        dilated_suc = [suc]
        dilated_suc += self.dilated_nbrs(suc, num_nodes, self.config['num_scales'])

        # --------------------build nodes left and right graph-----------------
        num_lanes = lane_idcs[-1].item() + 1

        left, right = dict(), dict()

        dist = np.expand_dims(ctrs, axis=1) - np.expand_dims(ctrs, axis=0)
        dist = np.sqrt((dist ** 2).sum(2))
        hi = np.arange(num_nodes).reshape(-1, 1).repeat(num_nodes, axis=1).reshape(-1)
        wi = np.arange(num_nodes).reshape(1, -1).repeat(num_nodes, axis=0).reshape(-1)
        row_idcs = np.arange(num_nodes)

        pre_mat = np.zeros((num_lanes, num_lanes))
        pre_mat[pre_pairs[:, 0], pre_pairs[:, 1]] = 1
        suc_mat = np.zeros((num_lanes, num_lanes))
        suc_mat[suc_pairs[:, 0], suc_pairs[:, 1]] = 1

        pairs = left_pairs
        if len(pairs) > 0:
            # construct lane left graph
            mat = np.zeros((num_lanes, num_lanes))
            mat[pairs[:, 0], pairs[:, 1]] = 1
            mat = (np.matmul(mat, pre_mat) + np.matmul(mat, suc_mat) + mat) > 0.5  # left lane's suc or pre lane is also self's left lane

            # filter with distance
            left_dist = dist.copy()
            # if lane j is the lane i's left, then all nodes in lane j is the left of any node in lane i
            mask = np.logical_not(mat[lane_idcs[hi], lane_idcs[wi]])
            # set the distance between nodes that has no left relation are very vert large
            left_dist[hi[mask], wi[mask]] = 1e6

            # find the each node's nearest node
            min_dist, min_idcs = left_dist.min(1), left_dist.argmin(1)
            # if nearest node's distance > self.config['cross_dist'], then this node does not have left node
            mask = min_dist < self.config['cross_dist']
            # if the angle between nearest node is too big , the this node does not have left node
            ui = row_idcs[mask]
            vi = min_idcs[mask]
            f1 = feats[ui]
            f2 = feats[vi]
            t1 = np.arctan2(f1[:, 1], f1[:, 0])
            t2 = np.arctan2(f2[:, 1], f2[:, 0])
            dt = np.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = np.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            left['u'] = ui.astype(np.int16)  # u is the idx of node that has left neighbor
            left['v'] = vi.astype(np.int16)  # v[i] is the idx of left neighbor of node u[i]
        else:
            left['u'] = np.zeros(0, np.int16)
            left['v'] = np.zeros(0, np.int16)

        pairs = right_pairs
        if len(pairs) > 0:
            mat = np.zeros((num_lanes, num_lanes))
            mat[pairs[:, 0], pairs[:, 1]] = 1
            mat = (np.matmul(mat, pre_mat) + np.matmul(mat, suc_mat) + mat) > 0.5

            right_dist = dist.copy()
            mask = np.logical_not(mat[lane_idcs[hi], lane_idcs[wi]])
            right_dist[hi[mask], wi[mask]] = 1e6

            min_dist, min_idcs = right_dist.min(1), right_dist.argmin(1)
            mask = min_dist < self.config['cross_dist']
            ui = row_idcs[mask]
            vi = min_idcs[mask]
            f1 = feats[ui]
            f2 = feats[vi]
            t1 = np.arctan2(f1[:, 1], f1[:, 0])
            t2 = np.arctan2(f2[:, 1], f2[:, 0])
            dt = np.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = np.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            right['u'] = ui.astype(np.int16)
            right['v'] = vi.astype(np.int16)
        else:
            right['u'] = np.zeros(0, np.int16)
            right['v'] = np.zeros(0, np.int16)

        graph = dict()
        graph['num_nodes'] = num_nodes
        # map node feats
        graph['ctrs'] = ctrs
        graph['feats'] = feats
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        # map node graph
        graph['pre'] = dilated_pre
        graph['suc'] = dilated_suc
        graph['left'] = left
        graph['right'] = right
        # lane pairs
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs

        return graph

'''
name; type; shape; meaning

suppose num of lanes is N, num of nodes is M

---------------map nodes level-------------
num_nodes: int; 1; num of nodes
=================feature===============
ctrs: ndarray; (M, 2); position
feats: ndarray; (M, 2); shape
turn: ndarray; (M, 2); turn type, [i, 0] = 1: left turn, [i, 1] = 1, right turn
control: ndarray; (M,); has control or not, [i] = 1, has contorl, [i] = 0, no control
intersect: ndarray; (M,); in intersect or not, [i] = 1, in, [i] = 0, not in
==================graph================
***************************************
pre: [dict]; (dilated neighbors adjacency matrix)
    pre[i]: dict neighors within 2^i step
        pre[i]['u']: ; array of nodes idx
        pre[i]['v']: ; array of nodes idx
    pre[i]['v'][j] is the pre within 2^i step neighbor node of pre[i]['u'][j]
***************************************
suc: [dict];
    suc[i]: dict neighors within 2^i step
        suc[i]['u']: ; array of nodes idx
        suc[i]['v']: ; array of nodes idx
    suc[i]['v'][j] is the suc within 2^i step neighbor node of suc[i]['u'][j]
***************************************
left: dict;
    left['u']; ndarray; (None,); array of nodes idx
    left['v']; ndarray; (None,); array of nodes idx
left['v'][i] is the left node of left['u'][i]
***************************************
right: dict;
    right['u']; ndarray; (None,); array of nodes idx
    right['v']; ndarray; (None,); array of nodes idx
right['v'][i] is the right node of right['u'][i]

---------------middle level-------------
lane_idcs: ndarray; (M,); [i] = n means node with id i belongs to lane with id n

---------------lane level---------------
pre_pairs; ndarray; (N, 2); [i, 1] is the pre lane of [i, 0]
suc_pairs; ndarray; (N, 2); [i, 1] is the suc lane of [i, 0]
left_pairs; ndarray; (N, 2); [i, 1] is the left lane of [i, 0]
right_pairs; ndarray; (N, 2); [i, 1] is the right lane of [i, 0]
'''