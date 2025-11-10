import copy

import torch


def default(net, contact, fl_coefficient, private_param, device):
    node_num = len(net)
    local_model = [{} for _ in range(node_num)]
    for n in range(node_num):
        nbr = contact[str(n)]  # the nodes n-th node contacted
        n_nbr = len(nbr)  # how many nodes n-th node contacted
        if n_nbr == 0:
            continue

        local_model[n] = net[n].state_dict()
        recv_models = []
        for k in nbr:
            param = copy.deepcopy(net[k].state_dict())
            recv_models.append(param)

        for k in range(n_nbr):
            for key in recv_models[k]:
                if key in private_param:
                    continue
                recv_models[k][key] = recv_models[k][key] - local_model[n][key]

        for k in range(n_nbr):
            for key in recv_models[k]:
                if key in private_param:
                    continue
                local_model[n][key] += (
                    recv_models[k][key] * fl_coefficient / float(n_nbr + 1)
                )

    # update nets
    for n in range(node_num):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            net[n].load_state_dict(local_model[n])


def update_nets(
    net,
    contact,
    fl_coefficient,
    private_param=[],
    device=torch.device("cpu"),
):
    default(net, contact, fl_coefficient, private_param, device)
