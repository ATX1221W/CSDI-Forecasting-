import torch, time, os, platform, sys
print('Python exe:', sys.executable)
print('Platform:', platform.platform())
print('Torch version:', torch.__version__)
print('Built with CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  [{i}]', torch.cuda.get_device_name(i))
    torch.cuda.synchronize()
    size = 8192
    a = torch.randn(size,size, device='cuda')
    b = torch.randn(size,size, device='cuda')
    torch.cuda.synchronize(); t=time.time(); c = a @ b; torch.cuda.synchronize()
    print('GPU matmul seconds:', round(time.time()-t,4))
    print('Result checksum:', float(c[0,0]))
else:
    print('GPU NOT AVAILABLE - This indicates a CPU-only Torch build or missing driver.')
    print('If driver exists (nvidia-smi works) reinstall torch with CUDA:')
    print('  conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia')
