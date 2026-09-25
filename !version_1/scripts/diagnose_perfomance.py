"""
Диагностика производительности и проверка GPU
"""

import torch
import time


def check_cuda():
    """Проверка CUDA и GPU"""
    print("=" * 70)
    print("CUDA & GPU DIAGNOSTICS")
    print("=" * 70)

    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   ✓ CUDA is available!")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     Name: {torch.cuda.get_device_name(i)}")
            print(f"     Compute Capability: {torch.cuda.get_device_capability(i)}")

            # Память
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            reserved_memory = torch.cuda.memory_reserved(i) / 1024 ** 3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024 ** 3

            print(f"     Total Memory: {total_memory:.2f} GB")
            print(f"     Reserved Memory: {reserved_memory:.2f} GB")
            print(f"     Allocated Memory: {allocated_memory:.2f} GB")
    else:
        print(f"   ✗ CUDA is NOT available!")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"\n   Possible reasons:")
        print(f"   1. PyTorch installed without CUDA support")
        print(f"   2. No NVIDIA GPU available")
        print(f"   3. CUDA drivers not installed")
        print(f"\n   To fix:")
        print(f"   - Check if you have NVIDIA GPU")
        print(f"   - Reinstall PyTorch with CUDA:")
        print(f"     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


def benchmark_cpu_vs_gpu():
    """Сравнение производительности CPU vs GPU"""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Создаем тестовые данные
    batch_size = 64
    seq_len = 120
    input_dim = 17

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input dim: {input_dim}")

    # Создаем фейковые данные
    x_cpu = torch.randn(batch_size, seq_len, input_dim)

    # CPU benchmark
    print(f"\n1. CPU Performance:")
    device_cpu = torch.device('cpu')
    x_cpu = x_cpu.to(device_cpu)

    start = time.time()
    for _ in range(100):
        _ = x_cpu @ torch.randn(input_dim, input_dim)
    cpu_time = time.time() - start

    print(f"   100 iterations: {cpu_time:.2f}s")
    print(f"   Speed: {100 / cpu_time:.1f} it/s")

    # GPU benchmark (если есть)
    if torch.cuda.is_available():
        print(f"\n2. GPU Performance:")
        device_gpu = torch.device('cuda')
        x_gpu = torch.randn(batch_size, seq_len, input_dim).to(device_gpu)

        # Прогрев GPU
        for _ in range(10):
            _ = x_gpu @ torch.randn(input_dim, input_dim, device=device_gpu)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            _ = x_gpu @ torch.randn(input_dim, input_dim, device=device_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"   100 iterations: {gpu_time:.2f}s")
        print(f"   Speed: {100 / gpu_time:.1f} it/s")
        print(f"\n   Speedup: {cpu_time / gpu_time:.1f}x faster than CPU")
    else:
        print(f"\n2. GPU Performance: SKIPPED (no CUDA)")


def check_dataloader_speed():
    """Проверка скорости DataLoader"""
    print("\n" + "=" * 70)
    print("DATALOADER SPEED TEST")
    print("=" * 70)

    try:
        from version_1.config.config import Config
        from version_1.data.dataset import create_dataloaders
        import pandas as pd

        config = Config()

        # Создаем фейковые данные
        print(f"\nCreating fake dataset...")
        n_samples = 5000
        data = {col: torch.randn(n_samples).numpy() for col in config.feature_columns + config.target_columns}
        df = pd.DataFrame(data)

        # Создаем датасет
        from version_1.data.dataset import VesselDataset
        dataset = VesselDataset(df, config, fit_scalers=True)

        # Тест с разными num_workers
        for num_workers in [0, 2, 4]:
            print(f"\nTesting with num_workers={num_workers}:")

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=num_workers,
                pin_memory=config.pin_memory and torch.cuda.is_available()
            )

            start = time.time()
            for i, (x, y) in enumerate(loader):
                if i >= 50:  # Первые 50 батчей
                    break
            elapsed = time.time() - start

            speed = 50 / elapsed
            print(f"   Speed: {speed:.1f} batch/s")

    except Exception as e:
        print(f"   Error: {e}")
        print(f"   Skipping DataLoader test")


def check_model_speed():
    """Проверка скорости модели"""
    print("\n" + "=" * 70)
    print("MODEL INFERENCE SPEED")
    print("=" * 70)

    try:
        from version_1.config.config import Config
        from version_1.models.vessel_predictor import VesselPredictor

        config = Config()

        # Создаем модель
        model = VesselPredictor(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            config=config
        )

        # Тестируем на CPU
        print(f"\n1. CPU Inference:")
        model_cpu = model.to('cpu')
        x_cpu = torch.randn(config.batch_size, config.sequence_length, config.input_dim)

        # Прогрев
        for _ in range(5):
            _ = model_cpu(x_cpu, target=None, teacher_forcing_ratio=0.0)

        start = time.time()
        for _ in range(20):
            _ = model_cpu(x_cpu, target=None, teacher_forcing_ratio=0.0)
        cpu_time = time.time() - start

        print(f"   20 iterations: {cpu_time:.2f}s")
        print(f"   Speed: {20 / cpu_time:.1f} batch/s")

        # Тестируем на GPU (если есть)
        if torch.cuda.is_available():
            print(f"\n2. GPU Inference:")
            model_gpu = model.to('cuda')
            x_gpu = x_cpu.to('cuda')

            # Прогрев
            for _ in range(5):
                _ = model_gpu(x_gpu, target=None, teacher_forcing_ratio=0.0)
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(20):
                _ = model_gpu(x_gpu, target=None, teacher_forcing_ratio=0.0)
            torch.cuda.synchronize()
            gpu_time = time.time() - start

            print(f"   20 iterations: {gpu_time:.2f}s")
            print(f"   Speed: {20 / gpu_time:.1f} batch/s")
            print(f"\n   Speedup: {cpu_time / gpu_time:.1f}x faster than CPU")
        else:
            print(f"\n2. GPU Inference: SKIPPED (no CUDA)")

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 70)
    print("PERFORMANCE DIAGNOSTICS TOOL")
    print("=" * 70)
    print("\nThis tool will help diagnose performance issues")
    print("Expected: ~40 batch/s with GPU, ~5 batch/s with CPU")
    print()

    # 1. Проверка CUDA
    check_cuda()

    # 2. Benchmark CPU vs GPU
    benchmark_cpu_vs_gpu()

    # 3. Проверка DataLoader
    check_dataloader_speed()

    # 4. Проверка модели
    check_model_speed()

    # Итоговые рекомендации
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  CRITICAL: CUDA is not available!")
        print("\nTo fix:")
        print("1. Check if you have NVIDIA GPU:")
        print("   - Right-click Desktop → Display settings → Advanced display")
        print("   - Or run: nvidia-smi (in terminal)")
        print("\n2. Install CUDA-enabled PyTorch:")
        print("   pip3 uninstall torch torchvision torchaudio")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n3. After installation, run this script again")
    else:
        print("\n✓ CUDA is available and working!")
        print("\nTips for maximum performance:")
        print("1. Use batch_size=64 or higher (if memory allows)")
        print("2. Set num_workers=2 or 4")
        print("3. Enable pin_memory=True")
        print("4. Enable cudnn.benchmark=True")
        print("\nExpected performance with GPU:")
        print("  - Training: 30-50 batch/s")
        print("  - Inference: 50-100 batch/s")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()