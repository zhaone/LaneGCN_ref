import networkx as nx
import numpy as np
from scipy import sparse
# TODO, speed up this func, ... too slow
def distance_dilate_v2(self, src_idcs, dest_idcs, nodes_feats):
    graph = nx.DiGraph()
    w = np.linalg.norm(nodes_feats, axis=1)
    # build graph
    init_edges = []
    for s, d in zip(src_idcs, dest_idcs):
        init_edges.append((s, d, w[s]))
    graph.add_weighted_edges_from(init_edges)
    # get shorted path
    shorted_path = nx.shortest_path(graph, weight='weight')  # maybe time consuming
    # build graph
    edges = {'src': [], 'dest': [], 'feats': []}
    for source, s_paths in shorted_path.items():
        for target, path in s_paths.items():
            path_len = len(path) - 2
            if path_len < 0:
                continue
            elif path_len == 0:
                dist_bt_pre_nodes = 0
            else:
                dist_bt_pre_nodes = w[path[1:-1]].sum()
                if dist_bt_pre_nodes > self.config['pre_suc_dist_thr']:
                    continue
            cosine = np.sum(nodes_feats[source] * nodes_feats[target]) / (w[source] * w[target])
            sine = np.cross(nodes_feats[source], nodes_feats[target]) / (w[source] * w[target])
            edges['src'].append(source)
            edges['dest'].append(target)
            edges['feats'].append([path_len, dist_bt_pre_nodes, cosine, sine])
    edges['src'] = np.array(edges['src']).astype(np.int)
    edges['dest'] = np.array(edges['dest']).astype(np.int)
    edges['feats'] = np.array(edges['feats']).astype(np.float)
    return edges

def distance_dilate_v3(self, src_idcs, dest_idcs, map_nodes_start, map_nodes_end, nodes_feats):
    num_nodes = len(nodes_feats)
    # get full connection relation graph (despite of distance)
    data = np.ones(len(src_idcs), np.bool)
    _csr = sparse.csr_matrix((data, (src_idcs, dest_idcs)), shape=(num_nodes, num_nodes))

    csr = _csr.copy()
    coo = csr.tocoo()
    ne = len(coo.row)

    iter_num = 0
    while True:
        iter_num += 1
        csr = csr*_csr + _csr
        coo = csr.tocoo()
        new_ne = len(coo.row)
        if new_ne == ne or iter_num > 4:
            break
        ne = new_ne

    new_src_idcs = coo.row
    new_dest_idcs = coo.col

    w = np.linalg.norm(nodes_feats, axis=1)
    final_src = []
    final_dest = []
    feats = []

    for source, target in zip(new_src_idcs, new_dest_idcs):
        dist = np.linalg.norm(map_nodes_start[source] - map_nodes_end[target])
        # filter with distance
        if dist > self.config['pre_suc_dist_thr']:
            continue
        final_src.append(source)
        final_dest.append(target)
        cosine = np.sum(nodes_feats[source]*nodes_feats[target]) / (w[source] * w[target])
        sine = np.cross(nodes_feats[source], nodes_feats[target]) / (w[source] * w[target])
        feats.append([dist, cosine, sine])

    edges = {
        'src': np.array(final_src).astype(np.int),
        'dest': np.array(final_dest).astype(np.int),
        'feats': np.array(feats).astype(np.float)
    }
    print(f'num of nodes: {num_nodes}')
    print(f'rank 0 num edge: {len(src_idcs)}')
    print(f'full num edge: {new_ne}')
    print(f'final num edge: {len(final_src)}')
    return edges