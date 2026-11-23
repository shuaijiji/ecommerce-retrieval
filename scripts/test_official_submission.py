import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import xml.etree.ElementTree as ET
from official_submission_common import (metadata,media_paths,select_pilot,validate_retrieval,validate_result)
from export_official_submission import write_xlsx,validate_xlsx,collect,NS
from rerank_pipeline import digest,save


def retrieval(q):
    ids=[str(i) for i in range(100,125)]
    route=dict(item_ids=ids,scores=[1-i*.01 for i in range(25)])
    return dict(photo_id=q,vision=route,multimodal=copy.deepcopy(route),seconds=.1)


def prediction(row,source_sha,expanded=True):
    ids=row['vision']['item_ids']
    ten=[dict(item_id=i,score=(.5 if expanded else .95)-j*.01) for j,i in enumerate(ids[:10])]
    new=[dict(item_id=i,score=.8-j*.01) for j,i in enumerate(ids[10:])] if expanded else []
    return dict(photo_id=row['photo_id'],retrieval_sha256=source_sha,top10_ranking=ten,
        ranking=sorted(ten+new,key=lambda r:(-r['score'],r['item_id'])),expanded=expanded,
        scored_new_items=len(new),seconds=.2,peak_memory_gib=4.5)


class OfficialSubmissionTest(unittest.TestCase):
    threshold=dict(cutoff=.87109375,tie_probability=.5,tie_seed=20261119)

    def test_cached_scores_and_routing_are_enforced(self):
        row=retrieval('00000000000000001')
        validate_retrieval(row,row['photo_id'],set(row['vision']['item_ids']))
        result=json.loads(json.dumps(prediction(row,'sha')))
        validate_result(result,row,self.threshold,'sha')
        result['ranking'][-1]['score']-=.001
        with self.assertRaises(ValueError):
            validate_result(result,row,self.threshold,'sha')
        result=prediction(row,'sha',expanded=False)
        validate_result(result,row,self.threshold,'sha')
        result['expanded']=True
        with self.assertRaises(ValueError):
            validate_result(result,row,self.threshold,'sha')

    def test_pool_mismatch_is_rejected(self):
        row=retrieval('1')
        with self.assertRaises(ValueError):
            validate_retrieval(row,'1',{'100'})

    def test_export_preserves_long_ids_and_leading_zeroes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'submission.xlsx'
            rows=[('00012345678901234567890',['999999999999999999999','100','101'])]
            write_xlsx(rows,path)
            validate_xlsx(path,rows)
            with zipfile.ZipFile(path) as archive:
                xml=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
                self.assertEqual(xml.findall(f'{{{NS}}}sheetData/{{{NS}}}row')[1][0].get('t'),'inlineStr')
            with self.assertRaises(ValueError):
                write_xlsx([('1',['100','100'])],path)

    def test_full_50k_export_size_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'full.xlsx'
            rows=[(str(10**18+i),['100','101','102']) for i in range(50000)]
            write_xlsx(rows,path)
            self.assertLessEqual(path.stat().st_size,10*1024**2)
            validate_xlsx(path,rows)

    def test_complete_export_collect_and_no_partial_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            data=root/'data'
            data.mkdir()
            photos=[dict(photo_id=q,photo_frames=['frame.jpg']) for q in ('1','2')]
            items=[dict(item_id=str(i),item_image='item.jpg') for i in range(100,125)]
            (data/'photo_info.jsonl').write_text('\n'.join(json.dumps(r) for r in photos))
            (data/'item_info.jsonl').write_text('\n'.join(json.dumps(r) for r in items))
            # The collection/inference path must never parse an answer file.
            (data/'label.json').write_text('intentionally invalid; must not be read')
            plan=dict(data_root=str(data),output_dir=str(root/'out'),threshold=self.threshold)
            for photo in photos:
                row=retrieval(photo['photo_id'])
                source=root/'out/retrieval'/(photo['photo_id']+'.json')
                save(row,source)
                save(prediction(row,digest(source)),root/'out/predictions'/(photo['photo_id']+'.json'))
            rows=collect(plan,expected=2)
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0][1],['110','111','112'])
            (root/'out/predictions/2.json').unlink()
            with self.assertRaisesRegex(ValueError,'coverage mismatch'):
                collect(plan,expected=2)

    def test_metadata_duplicate_and_media_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'photo_info.jsonl'
            p.write_text('{"photo_id":"1"}\n{"photo_id":"1"}\n')
            with self.assertRaises(ValueError):
                metadata(p,'photo_id')
            with self.assertRaises(ValueError):
                media_paths(root,{'1':dict(photo_id='1',photo_frames=['../outside.jpg'])},{})
            selected=select_pilot(['1','2','3'],2)
            self.assertEqual(selected,select_pilot(['3','1','2'],2))


if __name__=='__main__':
    unittest.main()
