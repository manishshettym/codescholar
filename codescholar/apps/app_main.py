import torch
from codescholar.apps.app_decl import scholarapp

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    scholarapp.run(host="0.0.0.0", port=3003)
