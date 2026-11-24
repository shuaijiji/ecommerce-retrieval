#!/usr/bin/env python3
"""Read-only XLSX format/provenance audit; optional exact comparison to server predictions."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import posixpath
import zipfile
import xml.etree.ElementTree as ET
from rerank_pipeline import digest,save

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def read_submission(path,expected=50000):
    if path.stat().st_size>10*1024**2:
        raise ValueError('file exceeds 10 MiB')
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('invalid ZIP')
        if any('vba' in name.lower() or 'externallink' in name.lower() for name in z.namelist()):
            raise ValueError('macro/external-link workbook')
        for name in z.namelist():
            if name.endswith('.rels'):
                if any(r.get('TargetMode')=='External' for r in ET.fromstring(z.read(name))):
                    raise ValueError('external relationship')
        workbook=ET.fromstring(z.read('xl/workbook.xml'))
        sheets=workbook.findall(f'{{{NS}}}sheets/{{{NS}}}sheet')
        if len(sheets)!=1 or sheets[0].get('state','visible')!='visible':
            raise ValueError('need exactly one visible sheet')
        relations={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        target=relations[sheets[0].get(f'{{{REL}}}id')]
        sheet_path=posixpath.normpath(target.lstrip('/') if target.startswith('/') else 'xl/'+target)
        if not sheet_path.startswith('xl/'):
            raise ValueError('unsafe sheet relationship')
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in si.iter(f'{{{NS}}}t')) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        result={}
        rows=0
        def cell_text(cell):
            if cell.find(f'{{{NS}}}f') is not None:
                raise ValueError('formula cell')
            kind=cell.get('t')
            if kind=='inlineStr':
                return ''.join(n.text or '' for n in cell.findall(f'.//{{{NS}}}t'))
            if kind=='s':
                return strings[int(cell.find(f'{{{NS}}}v').text)]
            raise ValueError('cells must contain text, not numeric IDs')
        with z.open(sheet_path) as stream:
            for _,row in ET.iterparse(stream,events=('end',)):
                if row.tag!=f'{{{NS}}}row':
                    continue
                cells=row.findall(f'{{{NS}}}c')
                if len(cells)!=2 or [c.get('r','').rstrip('0123456789') for c in cells]!=['A','B']:
                    raise ValueError('need exactly columns A/B, no blank or extra columns')
                values=[cell_text(c) for c in cells]
                if rows==0:
                    if values!=['photo_id','item_id_list']:
                        raise ValueError('incorrect column headers')
                else:
                    q,raw=values
                    if not q.isdigit() or q in result:
                        raise ValueError('invalid or duplicate video ID')
                    items=json.loads(raw)
                    if (not isinstance(items,list) or len(items)>3 or
                            not all(isinstance(i,str) and i.isdigit() for i in items) or len(set(items))!=len(items)):
                        raise ValueError('invalid item JSON string array')
                    result[q]=items
                rows+=1
                row.clear()
        if len(result)!=expected:
            raise ValueError(f'expected {expected} video rows, found {len(result)}')
    return result


def score_denominator_check(top1,top3,count):
    values={}
    for name,text in [('top1',top1),('top3',top3)]:
        value=Fraction(text)/100*count
        values[name]=dict(reported_percent=text,implied_correct=str(value),integer_correct_count=value.denominator==1)
    return dict(documented_count=count,checks=values,
        needs_organizer_clarification=not all(v['integer_correct_count'] for v in values.values()),
        conclusion='Screenshot scores alone do not establish actual scored subset, weighting, or denominator; do not infer 40k as fact')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xlsx',type=Path,required=True)
    p.add_argument('--official-root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--reported-top1',default='70.8750')
    p.add_argument('--reported-top3',default='82.9300')
    a=p.parse_args()
    rows=read_submission(a.xlsx)
    report=dict(xlsx=str(a.xlsx.resolve()),bytes=a.xlsx.stat().st_size,sha256=digest(a.xlsx),rows=len(rows),
        item_list_lengths=dict(Counter(len(v) for v in rows.values())),local_format_pass=True,
        official_prediction_match_verified=False,manifest_hash_match_verified=False,
        submission_receipt_verified=False,score_denominator=score_denominator_check(a.reported_top1,a.reported_top3,len(rows)))
    if a.official_root:
        from export_official_submission import collect
        from official_submission_common import assert_plan_files
        out=a.official_root.resolve()
        plan_path=out/'plan.json'
        plan=json.loads(plan_path.read_text())
        if Path(plan['output_dir']).resolve()!=out:
            raise ValueError('official output root differs from plan')
        assert_plan_files(plan)
        manifest=json.loads((out/'submission_top3.manifest.json').read_text())
        if (manifest['xlsx_sha256']!=report['sha256'] or manifest['rows']!=50000 or
                manifest['bytes']!=report['bytes'] or not manifest['local_format_validation'] or
                manifest['plan_sha256']!=digest(plan_path)):
            raise ValueError('XLSX differs from original manifest or frozen plan')
        expected=dict(collect(plan))
        if rows!=expected:
            raise ValueError('XLSX Top-3 differs from source predictions')
        report.update(official_prediction_match_verified=True,manifest_hash_match_verified=True,
            frozen_models=dict(embedding=plan['embedding_model'],reranker=plan['reranker_model']),threshold=plan['threshold'])
    if a.output:
        save(report,a.output)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
