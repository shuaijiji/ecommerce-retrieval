import json
from pathlib import Path
import tempfile
import unittest
from locate_evaluation_data import scan,inventory_file


class EvaluationInventoryTest(unittest.TestCase):
    def test_counts_do_not_certify_official_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            data=root/'evaluation'
            data.mkdir()
            (data/'photo_info.jsonl').write_text('\n'.join(json.dumps(dict(photo_id=str(i),photo_frames=['missing.jpg'])) for i in range(2)))
            (data/'item_info.jsonl').write_text(json.dumps(dict(item_id='p',item_image='missing.jpg'))+'\n')
            (data/'label.json').write_text('not read or parsed')
            report=scan(root,expected=2)['dataset_roots'][0]
            self.assertTrue(report['matches_expected_video_count_only'])
            self.assertFalse(report['official_dataset_identity_verified'])
            self.assertEqual(report['label_files_present'],['label.json'])
            self.assertFalse(report['photos']['sample_media'][0]['exists'])
            ignored=root/'artifacts'/'fake'
            ignored.mkdir(parents=True)
            (ignored/'photo_info.jsonl').write_text('invalid')
            (ignored/'item_info.jsonl').write_text('invalid')
            self.assertEqual(len(scan(root,expected=2)['dataset_roots']),1)

    def test_duplicate_and_unsafe_numeric_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'photo_info.jsonl'
            rows=[dict(photo_id=x,photo_frames=[]) for x in ('1','1',None,1.25,True)]
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            result=inventory_file(path,'photo_id','photo_frames')
            self.assertEqual(result['duplicate_ids'],1)
            self.assertEqual(result['invalid_ids'],3)
            self.assertEqual(result['missing_media_fields'],5)


if __name__=='__main__':
    unittest.main()
