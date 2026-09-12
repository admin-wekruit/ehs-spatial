"""Verify a restored platform database and every immutable blob before cutover.

Run under the destination PANOPTES_* configuration. Does not modify records.
Database dump/restore deliberately uses standard pg_dump/pg_restore tools.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ehs_spatial.platform.contracts import digest, validate_document
from ehs_spatial.platform.runtime import services


def verify(repository, blobs):
    counts = {'assets':0, 'revisions':0, 'publications':0}
    failures = []
    with repository._connect() as c:
        for row in c.execute('SELECT id,storage_key,sha256,size_bytes FROM assets'):
            try:
                blobs.get(row['storage_key'],row['sha256'].strip(),row['size_bytes'])
                counts['assets'] += 1
            except Exception:
                failures.append({'kind':'asset','id':str(row['id']),'code':'integrity_check_failed'})
        for row in c.execute('SELECT id,document,document_sha256 FROM scene_revisions'):
            try:
                validate_document(row['document'])
                if digest(row['document']) != row['document_sha256'].strip():
                    raise ValueError('document_hash_mismatch')
                counts['revisions'] += 1
            except Exception:
                failures.append({'kind':'revision','id':str(row['id']),'code':'document_check_failed'})
        for row in c.execute('SELECT p.id,p.scene_revision_id,p.snapshot,r.document_sha256 FROM publications p LEFT JOIN scene_revisions r ON r.id=p.scene_revision_id'):
            snapshot=row['snapshot'].get('revision',{})
            if str(row['scene_revision_id']) != snapshot.get('id') or not row['document_sha256'] or digest(snapshot.get('document')) != row['document_sha256'].strip():
                failures.append({'kind':'publication','id':str(row['id']),'code':'snapshot_check_failed'})
            else:
                counts['publications'] += 1
    return {'schemaVersion':1,'status':'passed' if not failures else 'failed','verified':counts,'failures':failures}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repository,blobs=services()
    result=verify(repository,blobs)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit(result['status']!='passed')


if __name__=='__main__': main()
