"""Read-only CUDA availability and sparse-gradient check for the remote PC."""
import json, platform
from pathlib import Path
import torch

report={'platform':platform.platform(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
if report['cuda_available']:
    report['gpu']=torch.cuda.get_device_name(0)
    report['capability']=list(torch.cuda.get_device_capability(0))
    report['memory_bytes']=torch.cuda.get_device_properties(0).total_memory
    try:
        matrix=torch.sparse_csr_tensor(torch.tensor([0,1,2],device='cuda'),torch.tensor([1,0],device='cuda'),torch.tensor([2.,3.],device='cuda'),size=(2,2))
        x=torch.tensor([[1.],[2.]],device='cuda',requires_grad=True)
        y=torch.sparse.mm(matrix,x)
        y.sum().backward()
        report['sparse_forward_ok']=bool(torch.allclose(y,torch.tensor([[4.],[3.]],device='cuda')))
        report['sparse_input_gradient_ok']=bool(torch.allclose(x.grad,torch.tensor([[3.],[2.]],device='cuda')))
    except Exception as error:
        report['sparse_error']=str(error)
Path('outputs/gpu-check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if not report.get('cuda_available') or not report.get('sparse_input_gradient_ok'):
    raise SystemExit('GPU verification incomplete. Do not start full training yet.')
