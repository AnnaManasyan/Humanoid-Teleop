"""Batch overlay: render internal_overlay.mp4 for every recording folder under a
parent directory.

  python batch_overlay.py <parent_folder>

Each subfolder that contains internal.mp4 + poses.txt + log.txt is processed with
the shared device calibration from overlay_hands.py. The video<->pose time offset
resets every recording session, so it is AUTO-ESTIMATED per folder (by matching the
projected hands to MediaPipe detections) instead of using the single-clip constant.
Needs hand_landmarker.task next to these scripts (see README).
"""
import os, sys, re, glob, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overlay_hands as O   # reuse calibration constants + helpers

def parse_log(p):
    ts=[int(m) for m in re.findall(r"First frame timestamp \(unix ms\):\s*(\d+)",open(p).read())]
    return float(ts[0]),float(ts[1])

def vpts(path):
    o=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
        "frame=pts_time","-of","csv=p=0",path],capture_output=True,text=True).stdout.split()
    return np.array([float(x) for x in o if x])

def proj(hands,nm,hp,hr):
    uv,_=cv2.projectPoints(O.hl(hands[nm],hp,hr).astype(np.float64),O.RVEC,O.TVEC,O.K,O.DIST)
    return uv.reshape(-1,2)

def best_offset(d,Tp,Tv,mpc,stab):
    """Offset (ms) minimising distance from projected hands to MediaPipe detections."""
    def med(off,step=3):
        ds=[]
        for n in range(0,len(Tv),step):
            if n>=len(mpc) or not mpc[n]: continue
            hp,hr,hands,k=O.pose_at(d,Tp,Tv[n]+off)
            for nm in ("left","right"):
                if nm not in hands or not(stab[nm][k] and stab[nm][k+1]): continue
                cen=proj(hands,nm,hp,hr); cen=cen[np.isfinite(cen).all(1)]
                if len(cen)<10: continue
                c=cen.mean(0); ds.append(min(np.linalg.norm(c-mc) for mc in mpc[n]))
        if not ds: return 1e9
        ds=np.array(ds); return np.median(ds[ds<np.percentile(ds,80)])
    coarse=min(np.arange(-120,521,20), key=med)
    fine=min(np.arange(coarse-20,coarse+21,5), key=med)
    return fine, med(fine)

def process(folder):
    vid=os.path.join(folder,O.VIDEO)
    vfirst,pfirst=parse_log(os.path.join(folder,O.LOG))
    d=O.parse_poses(os.path.join(folder,O.POSES)); Tp=pfirst+d["t_ms"]
    stab={}
    for nm in ("left","right"):
        tr=O.tracked_arr(d[nm]); d[nm][:,:,:3]=O.smooth_positions(d[nm],tr); stab[nm]=O.stability(d[nm],tr)
    Tv=vfirst+vpts(vid)*1000
    mpc=O.detect_hands(vid)
    off,fit=best_offset(d,Tp,Tv,mpc,stab)
    cap=cv2.VideoCapture(vid); fps=cap.get(cv2.CAP_PROP_FPS); W=int(cap.get(3)); H=int(cap.get(4))
    tmp=os.path.join(folder,"_tmp_overlay.mp4")
    vw=cv2.VideoWriter(tmp,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
    last={"left":-999,"right":-999}; n=0
    while True:
        ok,img=cap.read()
        if not ok: break
        if n<len(Tv):
            hp,hr,hands,k=O.pose_at(d,Tp,Tv[n]+off)
            cens=mpc[n] if n<len(mpc) else []
            for nm,col in [("left",(0,255,0)),("right",(0,128,255))]:
                if nm not in hands or not(stab[nm][k] and stab[nm][k+1]): continue
                uv=proj(hands,nm,hp,hr)
                if O.present(uv,cens,W,H): last[nm]=n
                elif n-last[nm]>O.HOLD: continue
                for a,b in O.BONES:
                    if O.okp(uv[a]) and O.okp(uv[b]): cv2.line(img,tuple(uv[a].astype(int)),tuple(uv[b].astype(int)),col,2,cv2.LINE_AA)
                for j in range(26):
                    if O.okp(uv[j]): cv2.circle(img,tuple(uv[j].astype(int)),3,col,-1,cv2.LINE_AA)
        vw.write(img); n+=1
    cap.release(); vw.release()
    out=os.path.join(folder,O.OUT)
    subprocess.run(["ffmpeg","-y","-v","error","-i",tmp,"-c:v","libx264","-pix_fmt","yuv420p","-crf","18",out])
    os.remove(tmp)
    print(f"  {os.path.basename(folder.rstrip('/'))}: offset={off:.0f}ms fit={fit:.0f}px -> {out} ({n} frames)")

def main():
    if len(sys.argv)<2:
        sys.exit("usage: python batch_overlay.py <parent_folder>")
    parent=sys.argv[1]
    folders=sorted(d for d in glob.glob(os.path.join(parent,"*"))
                   if os.path.isfile(os.path.join(d,"internal.mp4"))
                   and os.path.isfile(os.path.join(d,"poses.txt"))
                   and os.path.isfile(os.path.join(d,"log.txt")))
    print(f"processing {len(folders)} recording folder(s) under {parent}")
    for f in folders:
        try: process(f)
        except Exception as e: print(f"  {os.path.basename(f.rstrip('/'))}: FAILED ({e})")

if __name__=="__main__":
    main()
