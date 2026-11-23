#!/usr/bin/env python3
"""Validate every official prediction and export a two-column XLSX using standard OOXML."""
import argparse
import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
from official_submission_common import metadata,validate_retrieval,validate_result,assert_plan_files
from rerank_pipeline import digest,save

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'


def write_xlsx(rows,path):
    """Remote runtime exporter: inline strings preserve IDs, with no formulas/links/macros."""
    if not rows:
        raise ValueError('empty submission')
    cells=['<row r="1"><c r="A1" t="inlineStr"><is><t>photo_id</t></is></c><c r="B1" t="inlineStr"><is><t>item_id_list</t></is></c></row>']
    for index,(photo,items) in enumerate(rows,2):
        if not isinstance(photo,str) or not photo.isdigit() or not items or len(items)>3 or len(set(items))!=len(items) or not all(isinstance(i,str) and i.isdigit() for i in items):
            raise ValueError('invalid submission row')
        value=json.dumps(items,ensure_ascii=False,separators=(',',':'))
        cells.append(f'<row r="{index}"><c r="A{index}" s="1" t="inlineStr"><is><t>{escape(photo)}</t></is></c><c r="B{index}" s="1" t="inlineStr"><is><t>{escape(value)}</t></is></c></row>')
    sheet=f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{NS}"><cols><col min="1" max="1" width="22" customWidth="1"/><col min="2" max="2" width="70" customWidth="1"/></cols><sheetData>'+''.join(cells)+'</sheetData></worksheet>'
    parts={
        '[Content_Types].xml':'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>',
        '_rels/.rels':'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        'xl/workbook.xml':f'<?xml version="1.0"?><workbook xmlns="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="predictions" sheetId="1" state="visible" r:id="rId1"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels':'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
        'xl/styles.xml':f'<?xml version="1.0"?><styleSheet xmlns="{NS}"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="49" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>',
        'xl/worksheets/sheet1.xml':sheet}
    temporary=path.with_suffix('.xlsx.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,text in parts.items():
            archive.writestr(name,text)
    validate_xlsx(temporary,rows)
    temporary.replace(path)


def validate_xlsx(path,expected_rows):
    if path.stat().st_size>10*1024**2:
        raise ValueError('submission exceeds 10 MiB')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('damaged XLSX ZIP')
        workbook=ET.fromstring(archive.read('xl/workbook.xml'))
        sheets=workbook.findall(f'{{{NS}}}sheets/{{{NS}}}sheet')
        if len(sheets)!=1 or sheets[0].get('state','visible')!='visible':
            raise ValueError('submission must have one visible sheet')
        sheet=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        rows=sheet.findall(f'{{{NS}}}sheetData/{{{NS}}}row')
        expected=[('photo_id','item_id_list')]+[(q,json.dumps(ids,ensure_ascii=False,separators=(',',':'))) for q,ids in expected_rows]
        if len(rows)!=len(expected):
            raise ValueError('XLSX row count mismatch')
        for row,values in zip(rows,expected):
            cells=row.findall(f'{{{NS}}}c')
            if len(cells)!=2 or any(c.get('t')!='inlineStr' or c.find(f'{{{NS}}}f') is not None for c in cells):
                raise ValueError('non-text or formula cell in submission')
            actual=[c.find(f'{{{NS}}}is/{{{NS}}}t').text for c in cells]
            if actual!=list(values):
                raise ValueError('XLSX cell values changed during export')


def collect(plan,expected=50000):
    data=Path(plan['data_root'])
    out=Path(plan['output_dir'])
    photos=metadata(data/'photo_info.jsonl','photo_id')
    items=metadata(data/'item_info.jsonl','item_id')
    if len(photos)!=expected:
        raise ValueError('unexpected official photo count')
    paths={p.stem:p for p in (out/'predictions').glob('*.json')}
    if set(paths)!=set(photos):
        raise ValueError(f'prediction coverage mismatch: expected {len(photos)}, found {len(paths)}; no partial export')
    rows=[]
    item_set=set(items)
    for q in sorted(photos):
        source=out/'retrieval'/f'{q}.json'
        retrieved=json.loads(source.read_text())
        validate_retrieval(retrieved,q,item_set)
        result=json.loads(paths[q].read_text())
        validate_result(result,retrieved,plan['threshold'],digest(source))
        top3=[r['item_id'] for r in result['ranking'][:3]]
        rows.append((q,top3))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text())
    assert_plan_files(plan)
    rows=collect(plan)
    output=Path(plan['output_dir'])/'submission_top3.xlsx'
    write_xlsx(rows,output)
    save({'rows':len(rows),'xlsx_sha256':digest(output),'bytes':output.stat().st_size,
          'plan_sha256':digest(a.plan),'local_format_validation':True,
          'dataagent_format_validation':False,'submitted_to_organizer':False},output.with_suffix('.manifest.json'))
    print('Submission ready for DataAgent format validation: '+str(output))


if __name__=='__main__':
    main()
