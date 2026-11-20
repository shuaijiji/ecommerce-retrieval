#!/usr/bin/env python3
"""Stop only a verified embedding-evaluation launcher and its descendant processes (Linux)."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.pid <= 1:
        raise ValueError('invalid launcher PID')
    argv=Path(f'/proc/{a.pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    if not any(Path(v).name=='run_embedding_pilot_eval.sh' for v in argv):
        raise RuntimeError('PID is not the original embedding evaluation launcher; refusing to signal it')
    if not any(Path(v).resolve()==a.output_dir.resolve() for v in argv):
        raise RuntimeError('launcher output directory differs; refusing to signal it')
    os.kill(a.pid,signal.SIGSTOP)  # prevent advancing from item indexing to the next stage
    try:
        table=[]
        for line in subprocess.check_output(['ps','-eo','pid,ppid,stat'],text=True).splitlines()[1:]:
            pid,parent,state=line.split()
            table.append((int(pid),int(parent),state))
        targets={a.pid}
        while True:
            expanded=targets|{pid for pid,parent,_ in table if parent in targets}
            if expanded==targets:
                break
            targets=expanded
        for pid in sorted(targets,reverse=True):
            try:
                os.kill(pid,signal.SIGTERM)
                if pid==a.pid:
                    os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:
                pass
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            alive=[]
            for pid in targets:
                stat=Path(f'/proc/{pid}/stat')
                try:
                    state=stat.read_text().rsplit(')',1)[1].split()[0]
                    if state!='Z':
                        alive.append(pid)
                except FileNotFoundError:
                    pass
            if not alive:
                print('Verified launcher and workers stopped. Completed .pt shards preserved.')
                return
            time.sleep(.5)
        raise RuntimeError(f'processes still exiting: {alive}; do not launch another writer yet')
    finally:
        # Never leave an un-terminated verified launcher suspended after an error.
        try:
            os.kill(a.pid,signal.SIGCONT)
        except ProcessLookupError:
            pass


if __name__=='__main__':
    main()
