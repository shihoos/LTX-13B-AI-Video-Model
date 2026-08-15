import torch


def print_gpu_info() -> None:
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        print(
            f"GPU {i}: "
            f"{props.name} | "
            f"{props.total_memory / (1024**3):.2f} GB"
        )


if __name__ == "__main__":
    print_gpu_info()
