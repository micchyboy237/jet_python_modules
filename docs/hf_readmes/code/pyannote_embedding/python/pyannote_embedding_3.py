import torch
inference.to(torch.device("cuda"))
embedding = inference("audio.wav")