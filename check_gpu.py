import subprocess
import sys

print("=" * 55)
print("         GPU DIAGNOSTIC REPORT")
print("=" * 55)

# 1. Python & pip
print(f"\n[1] Python version: {sys.version}")

# 2. PyTorch
try:
    import torch
    print(f"\n[2] PyTorch version     : {torch.__version__}")
    print(f"    PyTorch CUDA version : {torch.version.cuda}")
    print(f"    CUDA available       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"    GPU name             : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"    VRAM                 : {vram:.1f} GB")
        # Quick tensor test
        x = torch.tensor([1.0]).cuda()
        print(f"    Tensor on GPU test   : ✅ PASSED ({x.device})")
    else:
        print("\n    ⚠️  CUDA NOT available — see diagnosis below")
except ImportError:
    print("\n[2] ❌ PyTorch is NOT installed")

# 3. nvidia-smi
print("\n[3] nvidia-smi output:")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        # Print just the key lines
        for line in result.stdout.splitlines()[:15]:
            print("   ", line)
    else:
        print("    ❌ nvidia-smi failed — no NVIDIA driver found")
except FileNotFoundError:
    print("    ❌ nvidia-smi not found — NVIDIA driver not installed")

# 4. Diagnosis
print("\n" + "=" * 55)
print("  DIAGNOSIS")
print("=" * 55)

try:
    import torch
    if torch.cuda.is_available():
        print("✅ GPU is fully working! Your app should use it.")
    elif torch.version.cuda is None:
        print("❌ CAUSE: PyTorch was installed WITHOUT CUDA support.")
        print("   (You have the CPU-only version of PyTorch)")
        print()
        print("   FIX — run ONE of these depending on your CUDA version:")
        print()
        print("   # CUDA 11.8:")
        print("   pip install torch torchvision torchaudio \\")
        print("     --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("   # CUDA 12.1:")
        print("   pip install torch torchvision torchaudio \\")
        print("     --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("   # CUDA 12.4:")
        print("   pip install torch torchvision torchaudio \\")
        print("     --index-url https://download.pytorch.org/whl/cu124")
        print()
        print("   Check your CUDA version first with: nvidia-smi")
    else:
        print("❌ CAUSE: PyTorch CUDA version does not match your driver.")
        print(f"   PyTorch expects CUDA {torch.version.cuda}")
        print("   but your driver may be different.")
        print("   Check: nvidia-smi — look for 'CUDA Version: X.X'")
        print("   Then reinstall PyTorch matching that version.")
except:
    pass

print("=" * 55)