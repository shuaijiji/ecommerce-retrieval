"""Conservative OCR/ASR exact-segment deduplication; no model, labels or heuristics."""
import re

FIELDS=('photo_ocr','photo_asr')


def dedup_segments(value):
    if not isinstance(value,str) or not value:
        return value
    # Do not split sentences, fuzzy-match, lowercase, or remove numeric/model tokens.
    parts=re.split(r'[|\r\n]+',value)
    seen=set()
    kept=[]
    removed=0
    for part in parts:
        key=re.sub(r'\s+',' ',part).strip()
        if not key:
            continue
        if key in seen:
            removed+=1
            continue
        seen.add(key)
        kept.append(part.strip())
    return '|'.join(kept) if removed else value


def transform_photo(photo):
    result=dict(photo)
    for field in FIELDS:
        if field in result:
            result[field]=dedup_segments(result[field])
    return result
