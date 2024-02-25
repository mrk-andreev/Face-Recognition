import torch


def main():
    model_alignment_path = './models/model_alignment.bin'
    device = 'cpu'

    model_align_faces = torch.load(model_alignment_path, map_location=torch.device(device))
    pass


if __name__ == '__main__':
    main()
