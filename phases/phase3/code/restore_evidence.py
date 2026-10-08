"""Restore exact historical evidence; no RNG, training, or network access."""
from pathlib import Path
import gzip,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def digest(b):return hashlib.sha256(b).hexdigest()
def load(name):
    target=ROOT/'results'/name
    if target.exists():return json.loads(target.read_text())
    archive=ROOT/'evidence'/(name+'.gz')
    return json.loads(gzip.decompress(archive.read_bytes()))
def main():
    manifest=json.loads((ROOT/'evidence/manifest.json').read_text())
    # Validate the entire request before writing any file.
    pending=[]
    for row in manifest['files']:
        archive=ROOT/row['archive'];compressed=archive.read_bytes()
        assert digest(compressed)==row['archive_sha256'],archive
        data=gzip.decompress(compressed)
        assert digest(data)==row['uncompressed_sha256'],archive
        assert len(data)==row['uncompressed_bytes'],archive
        target=ROOT/row['target']
        if target.exists():
            if target.read_bytes()!=data:raise SystemExit('Refusing to overwrite conflicting file: '+str(target))
        else:pending.append((target,data))
    for target,data in pending:
        target.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive creation also protects against a concurrent writer.
        with target.open('xb') as f:f.write(data)
    print('Verified',len(manifest['files']),'archives; restored',len(pending),'historical files.')
if __name__=='__main__':main()
