#!/usr/bin/env python3
"""Inventory local metadata counts; a 50k count alone does not certify the official dataset."""
import argparse
import json
import os
from pathlib import Path

PRUNE={'item_images','photo_frames','artifacts','models','.git','__pycache__','backups'}


def inventory_file(path,key,media_field,sample_count=10):
    seen=set()
    rows=duplicates=invalid=missing_media_fields=0
    samples=[]
    with path.open(encoding='utf-8') as handle:
        for number,line in enumerate(handle,1):
            if not line.strip():
                continue
            row=json.loads(line)
            if not isinstance(row,dict):
                raise ValueError(f'non-object metadata: {path}:{number}')
            rows+=1
            value=row.get(key)
            if isinstance(value,bool) or not isinstance(value,(str,int)) or str(value)=='':
                invalid+=1
            elif str(value) in seen:
                duplicates+=1
            else:
                seen.add(str(value))
            media=row.get(media_field)
            if not media:
                missing_media_fields+=1
            if len(samples)<sample_count:
                values=media if isinstance(media,list) else [media] if media else []
                samples.extend({'path':str(path.parent/relative),'exists':(path.parent/relative).is_file()}
                               for relative in values[:sample_count-len(samples)] if isinstance(relative,str))
    return dict(rows=rows,unique_ids=len(seen),duplicate_ids=duplicates,invalid_ids=invalid,
                missing_media_fields=missing_media_fields,sample_media=samples)


def scan(root,max_depth=6,expected=50000):
    if not root.is_dir():
        raise FileNotFoundError(root)
    datasets=[]
    for current,dirs,files in os.walk(root):
        folder=Path(current)
        depth=len(folder.relative_to(root).parts)
        dirs[:]=[] if depth>=max_depth else [name for name in dirs if name not in PRUNE]
        if not {'photo_info.jsonl','item_info.jsonl'}<=set(files):
            continue
        try:
            photos=inventory_file(folder/'photo_info.jsonl','photo_id','photo_frames')
            items=inventory_file(folder/'item_info.jsonl','item_id','item_image')
            datasets.append(dict(data_root=str(folder),photos=photos,items=items,
                label_files_present=[n for n in ('label.json','label.jsonl') if (folder/n).is_file()],
                matches_expected_video_count_only=(photos['rows']==expected and photos['unique_ids']==expected),
                official_dataset_identity_verified=False))
        except (ValueError,OSError,TypeError) as error:
            datasets.append(dict(data_root=str(folder),error=str(error),official_dataset_identity_verified=False))
    return dict(search_root=str(root),expected_official_video_count=expected,dataset_roots=datasets,
                limitation='counts and sample path existence only; no media decoding, identity certification, download or GPU inference')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-root',type=Path,default=Path('/code/ecommerce-retrieval/data'))
    p.add_argument('--max-depth',type=int,default=6)
    a=p.parse_args()
    print(json.dumps(scan(a.search_root.resolve(),a.max_depth),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
