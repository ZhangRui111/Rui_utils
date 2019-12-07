import torch


def index_by_array():
    """
    Array as index in torch.
    :return:
    """
    # ---------- 1 ----------#
    input_tensor = torch.randn((3, 3, 4))
    ind_tensor = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]).view((3, 3))
    print("input tensor with shape (3, 3, 4):")
    print(input_tensor)
    print("ind tensor with shape (3, 3):")
    print(ind_tensor)
    print("Pick elements in 3rd dim of input tensor by ind tensor:")
    # This request likes selective pick on 3rd dim following an index tensor.
    output_tensor = torch.gather(input_tensor, dim=2, index=ind_tensor[:, :, None])
    print("output tensor with shape (3, 3, 1):")
    print(output_tensor)

    # ---------- 2 ----------#
    input_tensor = torch.randn((3, 3))
    ind_tensor = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0]).view((3, 3)).byte()
    print("input tensor with shape (3, 3, 4):")
    print(input_tensor)
    print("ind tensor with shape (3, 3):")
    print(ind_tensor)
    print("Pick and set elements of input tensor by ind tensor with same shape:")
    # The ind tensor must have dtype=torch.uint8, i.e., .byte() instead of .float()
    output_tensor = input_tensor
    output_tensor[ind_tensor] = 0
    print("output tensor with shape (3, 3):")
    print(output_tensor)


def torch_logical_operations():
    """ torch logical operations: or, and, not, xor. """
    a = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0]).byte().view(3, 3)
    b = torch.tensor([0, 1, 1, 1, 1, 1, 0, 0, 0]).byte().view(3, 3)
    print("torch tensor or:", a | b)
    print("torch tensor and:", a & b)
    print("torch tensor not:", ~a)
    print("torch tensor xor:", a ^ b)


def get_index_by_value():
    """ get index by value. """
    a = torch.tensor([0, 1, 0, 1, 0, 1])
    print((a == 1).nonzero().squeeze())
    a = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0]).view(3, 3)
    print((a == 1).nonzero().squeeze())


def main():
    # ---------- design program to a special GPU ---------- #
    # 1. In shell,
    # CUDA_VISIBLE_DEVICES=0 python main.py
    # CUDA_VISIBLE_DEVICES=0,1 python main.py
    # or
    # 2. In python script,
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # ----------------------------------------------------- #
    index_by_array()
    # torch_logical_operations()
    # get_index_by_value()


if __name__ == '__main__':
    main()
