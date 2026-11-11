from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, platform, signal, subprocess, time

ROOT=Path('/workspace/genai_study')
OUT=ROOT/'artifacts/recovered-cpu-reader-path'
assert not OUT.exists(), 'preserve existing recovery output'
OUT.mkdir()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,x):
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
hist=json.loads((ROOT/'records/measurements/first-run-5000/execution.json').read_text())
sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/genai_study').glob('*.py')}
sources['requirements-lock.txt']=sha(ROOT/'requirements-lock.txt')
old_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'artifacts/first-run-5000-audit').glob('*/*') if p.is_file()}
assert len(old_files)==13
requests=[]
for old in hist['runs']:
    name=old['label'];command=[a.replace('artifacts/first-run-5000-audit/','artifacts/recovered-cpu-reader-path/') for a in old['command']]
    record=f'artifacts/recovered-cpu-reader-path/run-records/{name}'
    outer=['.venv/bin/python','-m','genai_study.run_record','--timeout','180','--output',record,'--']+command
    requests.append({'label':name,'command':command,'outer_argv':outer,'record_output':record,'output':f'artifacts/recovered-cpu-reader-path/{name}'})
plan={'created_utc':now(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'budget_seconds':600,'per_command_seconds':180,'start_reserve_seconds':185,'requests':requests,'source_sha256':sources,'historical_file_sha256':old_files,'hardware':{'cpu_model':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'affinity_count':len(os.sched_getaffinity(0)),'platform':platform.platform(),'python':platform.python_version(),'threads':1},'purpose':'Restore the documented 5000-step reader path on a recovered CPU instance; compare full arrays/checkpoint states to available historical output. No inference about method superiority or CPU causation.','concurrent_work':'One SD1.4 two-thread process plus documentation/browser work may run; shared host load and caches uncontrolled.','historical_source_note':'Historical checkpoint helper differs; compare current source identities and isolate numerical-path changes before interpretation.'}
atomic(OUT/'plan.json',plan)
state={'started_utc':now(),'status':'running','requests':[dict(r,status='not_attempted') for r in requests],'stop_reason':None,'driver_pid':os.getpid()}
atomic(OUT/'execution.json',state)
started=time.monotonic(); deadline=started+600; cancel={'signal':None}; active=None
for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda sig,frame:cancel.update(signal=sig))
try:
    for r in state['requests']:
        if deadline-time.monotonic()<185 or cancel['signal']:
            state['stop_reason']='budget_or_signal_before_launch';break
        assert all(sha(ROOT/p)==h for p,h in sources.items())
        assert not (ROOT/r['output']).exists() and not (ROOT/r['record_output']).exists()
        r.update(status='running',started_utc=now())
        with (OUT/(r['label']+'-recorder.txt')).open('x') as log:
            active=subprocess.Popen(r['outer_argv'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            r['pid']=active.pid;atomic(OUT/'execution.json',state)
            print('started',r['label'],active.pid,flush=True)
            while active.poll() is None:
                if cancel['signal'] or time.monotonic()>=deadline-5:
                    active.terminate()
                    try:active.wait(timeout=5);r['cleanup_confirmed']=True
                    except subprocess.TimeoutExpired:r['cleanup_confirmed']=False
                    state['stop_reason']='signal_or_campaign_deadline';break
                time.sleep(.1)
        r.update(returncode=active.poll(),finished_utc=now())
        if active.poll() is None:
            r['status']='cleanup_unconfirmed';break
        rec=json.loads((ROOT/r['record_output']/'run.json').read_text())
        r.update(status=rec['status'],wall_seconds=rec['wall_seconds'],exit_code=rec['exit_code'])
        atomic(OUT/'execution.json',state);print('finished',r['label'],r['status'],r['wall_seconds'],flush=True)
        active=None
        if r['status']!='succeeded':state['stop_reason']='command_not_succeeded';break
    state['status']='completed' if all(r['status']=='succeeded' for r in state['requests']) else 'stopped'
finally:
    if active is not None and active.poll() is None and not any(r.get('cleanup_confirmed') is False for r in state['requests']):
        active.terminate()
        try:active.wait(timeout=5)
        except subprocess.TimeoutExpired:state['cleanup_unconfirmed_pid']=active.pid
    state.update(finished_utc=now(),campaign_wall_seconds=time.monotonic()-started)
    for r in state['requests']:
        if r['status']=='not_attempted':r['reason']=state['stop_reason']
    state['historical_files_preserved']=all((ROOT/p).is_file() and sha(ROOT/p)==h for p,h in old_files.items())
    state['source_files_preserved']=all(sha(ROOT/p)==h for p,h in sources.items())
    atomic(OUT/'execution.json',state)
print(json.dumps({'status':state['status'],'wall':state['campaign_wall_seconds'],'historical_files_preserved':state['historical_files_preserved']}),flush=True)
raise SystemExit(0 if state['status']=='completed' else 1)
