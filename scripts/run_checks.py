"""One consolidated review runner; diagnostics stay outside the research project."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

PACKAGE=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--out-parent',type=Path,required=True)
    args=parser.parse_args()
    out_parent=args.out_parent.resolve()
    if args.project and out_parent.is_relative_to(args.project.resolve()):
        parser.error('Output directory must be outside the project')
    out=out_parent/('BGREL_R5_REVIEW_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',BGREL_CODE=str(PACKAGE/'code'))
    env['PYTHONPATH']=str(PACKAGE/'code')+os.pathsep+env.get('PYTHONPATH','')
    results={}
    def command(name,argv,timeout):
        print('Running '+name,flush=True)
        try:
            with (out/(name+'.log')).open('w',encoding='utf-8') as f:
                r=subprocess.run(argv,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
            results[name]={'status':'PASS' if r.returncode==0 else 'FAIL','exit_code':r.returncode}
        except Exception as exc:
            results[name]={'status':'FAIL','error':str(exc)}
            with (out/(name+'.log')).open('a',encoding='utf-8') as f:f.write('\n'+str(exc))
    command('regression',[sys.executable,'-m','unittest','discover','-s',str(PACKAGE/'tests'),'-v'],180)
    # --help exits before phase code: no preflight, train, overwrite or TEST access.
    entries=[]
    for p in sorted((PACKAGE/'code').glob('*.py')):
        tree=ast.parse(p.read_text())
        if any(isinstance(n,ast.If) and '__main__' in ast.unparse(n.test) for n in tree.body):entries.append(p)
    for p in entries:command('cli_'+p.stem,[sys.executable,str(p),'--help'],60)
    command('transaction_regression',[sys.executable,str(PACKAGE/'scripts/source_transaction.py'),'simulate'],60)
    command('workspace_transaction_simulation',[sys.executable,str(PACKAGE/'scripts/workspace_transaction.py'),'simulate'],60)
    if args.project:
        command('baseline_snapshot',[sys.executable,str(PACKAGE/'scripts/check_b0_snapshot.py'),'--project-root',str(args.project.resolve())],120)
        command('project_review',[sys.executable,str(PACKAGE/'scripts/review_project.py'),
                '--project',str(args.project.resolve()),'--out',str(out)],900)
    command('environment',[sys.executable,'-m','pip','freeze'],60)
    manifest={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in out.rglob('*') if p.is_file()}
    summary={'status':'CHECKS_PASS' if all(v['status']=='PASS' for v in results.values()) else 'CHECKS_HAVE_FAILURES',
             'results':results,'file_sha256':manifest,'deployment_status':'NOT_DEPLOYED',
             'project_access':'READ_ONLY' if args.project else 'NO_PROJECT_ACCESS',
             'test_pixels_opened':False}
    (out/'CHECK_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    print('REVIEW_ZIP: '+str(archive),flush=True)
    print(summary['status']+'; NOT_DEPLOYED; TEST_PIXELS_OPENED=False',flush=True)
    return 0 if summary['status']=='CHECKS_PASS' else 1
if __name__=='__main__':sys.exit(main())
