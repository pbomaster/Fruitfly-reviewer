"""Windows checkpoint/pause control without changing script execution policy."""
import argparse,ctypes,datetime,json,subprocess,sys
from pathlib import Path

RUNNERS={'sections-v9':'train_sections.py','coverage-v8':'train_coverage.py','transition-v7':'train_transition.py','sequential-v6':'train_sequential.py'}

def alive(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:
        if ctypes.get_last_error()==87: return False
        raise RuntimeError('Cannot verify process; refusing duplicate launch.')
    try:
        code=ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)): raise RuntimeError('Cannot read process state.')
        return code.value==259
    finally: kernel.CloseHandle(handle)

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['status','pause','resume'])
    p.add_argument('--run',choices=RUNNERS,default='coverage-v8'); args=p.parse_args()
    root=Path(__file__).resolve().parent.parent; out=root/'outputs'/args.run
    state=json.loads((out/'status.json').read_text(encoding='utf8'))
    running=alive(state['pid']); pause=out/'PAUSE.request'
    if args.action=='status':
        print(json.dumps({**state,'process_alive':running,'checkpointed_and_process_stopped':not running and state['state'] in ['paused','completed']},indent=2)); return
    if args.action=='pause':
        if running:
            pause.write_text('Checkpoint and pause requested by local control.',encoding='utf8')
            print('Pause requested. Check status for paused state and stopped process before shutdown.')
        else: print('Process is already stopped; saved checkpoint remains available.')
        return
    if running: raise RuntimeError('A process with the saved PID is still alive; refusing duplicate launch.')
    if state['state']=='completed': raise RuntimeError('Planned training already completed. Do not repeat it blindly.')
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    if (9,25)<=(now.hour,now.minute)<(19,30): raise RuntimeError('Daytime shutdown window; resume after 19:30 Korea time.')
    if pause.exists(): pause.unlink()
    subprocess.run([sys.executable,str(root/'work'/RUNNERS[args.run]),'--resume'],cwd=root,check=True)

if __name__=='__main__': main()
