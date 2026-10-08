"""Run one existing frozen config with an exclusive raw log and visible progress."""
from pathlib import Path
import argparse, subprocess, sys, os, signal

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args()
    out=a.output_dir.resolve(); log=out.with_name(out.name+'.stdout.log')
    if out.exists(): raise FileExistsError(out)
    log.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-B',str(Path(__file__).with_name('frontier.py')),'--config',str(a.config.resolve()),'--output-dir',str(out),'--workers',str(a.workers)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with log.open('x') as stream:
        stream.write('COMMAND '+repr(command)+'\n');stream.flush()
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,start_new_session=True)
        try:
            for line in proc.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            status=proc.wait()
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
            raise
    return status

if __name__=='__main__':sys.exit(main())
