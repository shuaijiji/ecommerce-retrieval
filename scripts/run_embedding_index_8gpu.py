#!/usr/bin/env python3
"""Resume existing item shards; require measured 1/2/4/8-GPU scaling before full encoding."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time


def summarize(records, samples, uuids):
    if len(records) != len(uuids) or any(r['encoded_items'] <= 0 for r in records):
        raise ValueError('incomplete performance results')
    start = min(r['start_unix'] for r in records)
    end = max(r['end_unix'] for r in records)
    rate = sum(r['encoded_items'] for r in records) / (end - start)
    medians = {}
    for uuid, record in zip(uuids, records):
        values = [s[uuid] for s in samples if record['start_unix'] <= s['timestamp'] <= record['end_unix']]
        if not values:
            raise ValueError(f'no encoding-window utilization samples for {uuid}')
        medians[uuid] = statistics.median(values)
    return {'items_per_second': rate, 'gpu_utilization_medians': medians,
            'encoded_items': sum(r['encoded_items'] for r in records)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/code/ecommerce-retrieval'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a = p.parse_args()
    root, model, output = a.root.resolve(), a.model.resolve(), a.output_dir.resolve()
    physical = a.gpus.split(',')
    if len(physical) != 8 or len(set(physical)) != 8:
        raise ValueError('specify eight distinct physical GPU indices')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'index_8gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # The old shell launcher does not acquire this lock. Reject any existing index writer.
    for line in subprocess.check_output(['ps','-eo','pid,args'], text=True).splitlines():
        if 'build_item_index.py' not in line:
            continue
        try:
            argv = shlex.split(line)
            target = Path(argv[argv.index('--output-dir')+1]).resolve()
        except (ValueError, IndexError):
            continue
        if target in (output/'train_item_index',output/'dev_item_index'):
            raise RuntimeError('old index worker still running; stop its launcher and worker before switching')
    uuids = []
    for gpu in physical:
        row = subprocess.check_output(['nvidia-smi','--id='+gpu,
            '--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'], text=True).strip().split(',')
        uuid, used = row[0].strip(), int(row[1])
        if used > 1024:
            raise RuntimeError(f'GPU {gpu} occupied ({used} MiB); no busy-GPU override')
        uuids.append(uuid)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    logdir = output/'logs'/('eight_gpu_'+stamp)
    logdir.mkdir(parents=True, exist_ok=False)
    data = root/'data/train/all_data_package'
    split = root/'artifacts/train_v0/split_quarantine_duplicates'
    script = root/'scripts/build_item_index.py'

    def run_group(side, count, benchmark=False):
        label = f'{side}_{count}gpu_' + ('perf' if benchmark else 'full')
        children, handles, reports = [], [], []
        samples = []
        try:
            for worker in range(count):
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=uuids[worker], OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
                cmd = ['python','-u',str(script),'--model',str(model),'--record-weight-hashes',
                       '--data-root',str(data),'--item-ids',str(split/f'{side}_item_ids.txt'),
                       '--output-dir',str(output/f'{side}_item_index'),
                       '--partitions',str(count),'--partition-index',str(worker)]
                if benchmark:
                    report = logdir/f'{label}_{worker}.json'
                    reports.append(report)
                    cmd += ['--max-new-shards','4','--benchmark-output',str(report)]
                handle = (logdir/f'{label}_{worker}.log').open('w')
                handles.append(handle)
                children.append(subprocess.Popen(cmd,cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT))
            print(f'{label} launched; logs: {logdir}', flush=True)
            with (logdir/f'{label}_gpu.csv').open('w') as trace:
                writer = csv.writer(trace)
                writer.writerow(['unix_time','uuid','gpu_utilization','memory_used_mib','power_watts'])
                while any(child.poll() is None for child in children):
                    if any(child.poll() not in (None,0) for child in children):
                        raise RuntimeError(f'{label} worker failed; inspect logs')
                    text = subprocess.check_output(['nvidia-smi','--id='+','.join(uuids[:count]),
                        '--query-gpu=uuid,utilization.gpu,memory.used,power.draw',
                        '--format=csv,noheader,nounits'],text=True)
                    sample = {'timestamp': time.time()}
                    for row in csv.reader(text.splitlines()):
                        uuid = row[0].strip()
                        sample[uuid] = float(row[1])
                        writer.writerow([sample['timestamp'], *[v.strip() for v in row]])
                    samples.append(sample)
                    trace.flush()
                    time.sleep(2)
            if any(child.returncode for child in children):
                raise RuntimeError(f'{label} failed')
            if benchmark:
                return summarize([json.loads(path.read_text()) for path in reports],samples,uuids[:count])
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            for handle in handles:
                handle.close()

    # Performance stages write real, non-overlapping missing shards into the same index.
    # Batch, model and encoding protocol stay unchanged; these shards are reused afterwards.
    metrics = {}
    for n in (1,2,4,8):
        metrics[str(n)] = run_group('train',n,benchmark=True)
        print(json.dumps({str(n):metrics[str(n)]}),flush=True)
    speedup = metrics['8']['items_per_second']/metrics['1']['items_per_second']
    passed = speedup >= 4 and all(v>=60 for v in metrics['8']['gpu_utilization_medians'].values())
    report = {'metrics':metrics,'eight_gpu_speedup':speedup,'performance_gate':passed}
    (logdir/'performance_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed:
        raise RuntimeError(f'8-GPU performance gate failed; preserve shards and inspect {logdir}/performance_summary.json')
    for side in ('train','dev'):
        run_group(side,8)
        # Finalize by validating all expected shards, without re-encoding completed work.
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = uuids[0]
        subprocess.run(['python','-u',str(script),'--model',str(model),'--record-weight-hashes',
            '--data-root',str(data),'--item-ids',str(split/f'{side}_item_ids.txt'),
            '--output-dir',str(output/f'{side}_item_index')],cwd=root,env=env,check=True)
    print('8-GPU item indexes complete; continuing fixed 2,000-query evaluation on one GPU',flush=True)
    env = os.environ.copy()
    env['EVAL_GPU'] = physical[0]
    subprocess.run(['bash',str(root/'scripts/run_embedding_pilot_eval.sh'),str(root),
                    str(model.parent),str(output)],cwd=root,env=env,check=True)


if __name__ == '__main__':
    main()
