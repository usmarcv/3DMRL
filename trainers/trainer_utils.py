import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn



def merge_results_dist(part_logits, part_labels):

    part_logits = torch.cat(part_logits, dim=0)
    part_labels = torch.cat(part_labels, dim=0)

    if not dist.is_available() or not dist.is_initialized():
        return part_logits.cpu(), part_labels.cpu()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logits_list = [torch.zeros_like(part_logits) for _ in range(world_size)]
    labels_list = [torch.zeros_like(part_labels) for _ in range(world_size)]

    dist.all_gather(logits_list, part_logits)
    dist.all_gather(labels_list, part_labels)

    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    if rank == 0:
        return logits_all.cpu(), labels_all.cpu()
    else:
        return None, None



def merge_two_branch_results_dist(part_image_logits, part_text_logits, part_labels):

    part_image_logits = torch.cat(part_image_logits, dim=0)
    part_text_logits = torch.cat(part_text_logits, dim=0)
    part_labels = torch.cat(part_labels, dim=0)

    gather_image = [torch.zeros_like(part_image_logits) for _ in range(dist.get_world_size())]
    gather_text = [torch.zeros_like(part_text_logits) for _ in range(dist.get_world_size())]
    gather_labels = [torch.zeros_like(part_labels) for _ in range(dist.get_world_size())]

    dist.all_gather(gather_image, part_image_logits)
    dist.all_gather(gather_text, part_text_logits)
    dist.all_gather(gather_labels, part_labels)

    logits_image = torch.cat(gather_image, dim=0)
    logits_text = torch.cat(gather_text, dim=0)
    labels_all = torch.cat(gather_labels, dim=0)

    return logits_image, logits_text, labels_all


# def merge_two_branch_results_dist(tmpdir, part_image_logits, part_text_logits, part_labels):
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()

#     os.makedirs(tmpdir, exist_ok=True)
#     pickle.dump(torch.cat(part_image_logits).cpu().numpy(),
#                 open(os.path.join(tmpdir, 'result_part_image_{}.pkl'.format(rank)), 'wb'))
#     pickle.dump(torch.cat(part_text_logits).cpu().numpy(),
#                 open(os.path.join(tmpdir, 'result_part_text_{}.pkl'.format(rank)), 'wb'))
#     pickle.dump(torch.cat(part_labels).cpu().numpy(),
#                 open(os.path.join(tmpdir, 'label_part_{}.pkl'.format(rank)), 'wb'))
    
#     dist.barrier()
#     if rank == 0:
#         part_image_list = []
#         part_text_list = []
#         part_label_list = []

#         for i in range(world_size):
#             part_image_file = os.path.join(tmpdir, 'result_part_image_{}.pkl'.format(i))
#             part_text_file = os.path.join(tmpdir, 'result_part_text_{}.pkl'.format(i))
#             part_label = os.path.join(tmpdir, 'label_part_{}.pkl'.format(i))

#             part_image_list.append(pickle.load(open(part_image_file, 'rb')))
#             part_text_list.append(pickle.load(open(part_text_file, 'rb')))
#             part_label_list.append(pickle.load(open(part_label, 'rb')))

#         part_image_list = np.concatenate(part_image_list, axis=0)
#         part_text_list = np.concatenate(part_text_list, axis=0)
#         part_label_list = np.concatenate(part_label_list, axis=0)

#         logits_image = torch.from_numpy(part_image_list)
#         logits_text = torch.from_numpy(part_text_list)
#         labels_all = torch.from_numpy(part_label_list)

#         return logits_image, logits_text, labels_all
#     else:
#         return None, None, None