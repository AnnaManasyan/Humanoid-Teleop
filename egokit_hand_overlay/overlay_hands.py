"""Paint Quest-3 OpenXR hand tracking onto the EgoKit ego video.

Self-contained: given internal.mp4 + poses.txt + log.txt in this folder, it
detects hands with MediaPipe (for the presence gate), projects the 26-joint
OpenXR hands using the device calibration below, and writes internal_overlay.mp4.

Pipeline: device intrinsics + lens distortion (from a checkerboard), a camera-
from-head extrinsic, and a fixed video<->pose time offset; with temporal
smoothing, a tracking-stability gate, and a MediaPipe presence gate so a hand is
only drawn when it is genuinely visible (not when tracked-but-out-of-view or
hallucinated).

Run:  .venv/bin/python overlay_hands.py
Needs: numpy, opencv-python, mediapipe, and hand_landmarker.task in this folder.
"""
import os, re, subprocess, numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks import python as mpy
from mediapipe.tasks.python import vision

# ---- files ----
VIDEO="internal.mp4"; POSES="poses.txt"; LOG="log.txt"; OUT="internal_overlay.mp4"
MODEL=os.path.join(os.path.dirname(os.path.abspath(__file__)),"hand_landmarker.task")

# ---- device calibration (intrinsics + distortion + camera-from-head extrinsic) ----
FX=FY=885.8618993046906
CX=647.7701577782246; CY=384.4103345688281
DIST=np.array([-0.056695879105550975,0.17051453060525537,0.020500780201544225,
               0.0020440962292691527,-0.3092042069509731])
RVEC=np.array([0.2348113665100521,-0.018848236258459803,-0.006427630453090796])
TVEC=np.array([0.029267716235218514,0.005918057395272774,-0.002785470379678182])
FLIP=np.array([1.,-1.,1.])            # Unity head-local -> OpenCV camera axes
OFFSET=385.0                          # ms, video -> pose time offset (this recording)
K=np.array([[FX,0,CX],[0,FY,CY],[0,0,1.0]])

# ---- tuning ----
DEBOUNCE=3; MAXJUMP=0.06; SMOOTH=2    # stability + smoothing (pose frames)
PRESENCE_PX=190; MARGIN=50; HOLD=2    # MediaPipe presence gate

# OpenXR 26-joint skeleton bones (wrist=1; fingers thumb..little)
BONES=[(1,2),(2,3),(3,4),(4,5),(1,6),(6,7),(7,8),(8,9),(9,10),(1,11),(11,12),
       (12,13),(13,14),(14,15),(1,16),(16,17),(17,18),(18,19),(19,20),(1,21),
       (21,22),(22,23),(23,24),(24,25)]

def quat_to_R(q):
    x,y,z,w=q; n=x*x+y*y+z*z+w*w
    if n<1e-12: return np.eye(3)
    s=2.0/n
    return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
                     [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
                     [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])

def parse_poses(path):
    tm,hp,hr,L,R=[],[],[],[],[]
    for line in open(path):
        if line.startswith("#") or not line.strip(): continue
        v=line.split()
        if len(v)<373: continue
        f=list(map(float,v)); tm.append(f[1]); hp.append(f[2:5]); hr.append(f[5:9])
        L.append(np.array(f[9:9+26*7]).reshape(26,7))
        R.append(np.array(f[9+26*7:9+52*7]).reshape(26,7))
    return {"t_ms":np.array(tm),"head_pos":np.array(hp),"head_rot":np.array(hr),
            "left":np.array(L),"right":np.array(R)}

def quat_slerp(q0,q1,a):
    d=np.dot(q0,q1)
    if d<0: q1=-q1; d=-d
    if d>0.9995: q=q0+a*(q1-q0); return q/np.linalg.norm(q)
    th=np.arccos(d)*a; q2=q1-q0*d; q2/=np.linalg.norm(q2)
    return q0*np.cos(th)+q2*np.sin(th)

def tracked1(j): return np.linalg.norm(j[:,:3]-j[1:2,:3],axis=1).sum()>0.05

def pose_at(d,Tp,t):
    k=int(np.clip(np.searchsorted(Tp,t)-1,0,len(Tp)-2))
    a=np.clip((t-Tp[k])/(Tp[k+1]-Tp[k]+1e-9),0,1)
    hp=(1-a)*d["head_pos"][k]+a*d["head_pos"][k+1]
    hr=quat_slerp(d["head_rot"][k],d["head_rot"][k+1],a); hands={}
    for nm in ("left","right"):
        j0,j1=d[nm][k],d[nm][k+1]
        if tracked1(j0) and tracked1(j1): hands[nm]=(1-a)*j0[:,:3]+a*j1[:,:3]
        elif tracked1(j0): hands[nm]=j0[:,:3]
        elif tracked1(j1): hands[nm]=j1[:,:3]
    return hp,hr,hands,k

def tracked_arr(a): return np.array([tracked1(a[i]) for i in range(len(a))])

def smooth_positions(a,tr,W=SMOOTH):
    N=len(a); sm=a[:,:,:3].astype(np.float64).copy()
    for i in range(N):
        if not tr[i]: continue
        lo=hi=i
        while lo>i-W and lo-1>=0 and tr[lo-1]: lo-=1
        while hi<i+W and hi+1<N and tr[hi+1]: hi+=1
        sm[i]=a[lo:hi+1,:,:3].mean(axis=0)
    return sm

def stability(a,tr):
    disp=np.r_[0.0,np.max(np.linalg.norm(a[1:,:,:3]-a[:-1,:,:3],axis=2),axis=1)]
    st=tr.copy()
    for k in range(1,DEBOUNCE): st[k:]&=tr[:-k]
    return st & (disp<MAXJUMP)

def hl(P,hp,hr): return ((P-hp)@quat_to_R(hr))*FLIP
def okp(p): return np.all(np.isfinite(p)) and np.all(np.abs(p)<1e5)

def parse_log(path):
    ts=[int(m) for m in re.findall(r"First frame timestamp \(unix ms\):\s*(\d+)",open(path).read())]
    return float(ts[0]),float(ts[1])

def video_pts(path):
    o=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
        "frame=pts_time","-of","csv=p=0",path],capture_output=True,text=True).stdout.split()
    return np.array([float(x) for x in o if x])

def detect_hands(path):
    lm=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpy.BaseOptions(model_asset_path=MODEL),num_hands=2,
        min_hand_detection_confidence=0.4,min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,running_mode=vision.RunningMode.VIDEO))
    cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS) or 59.43; cen=[]; n=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        H,W=fr.shape[:2]
        r=lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)),int(n*1000/fps))
        cen.append([np.mean([[p.x*W,p.y*H] for p in hh],axis=0)
                    for hh,sc in zip(r.hand_landmarks,r.handedness) if sc[0].score>=0.5]); n+=1
    cap.release(); return cen

def present(uv,cens,W,H):
    c=uv[np.isfinite(uv).all(1)]
    if len(c)<10: return False
    cen=c.mean(0)
    if not(-MARGIN<=cen[0]<=W+MARGIN and -MARGIN<=cen[1]<=H+MARGIN): return False
    if not cens: return False
    return min(np.linalg.norm(cen-mc) for mc in cens)<PRESENCE_PX

def main():
    vfirst,pfirst=parse_log(LOG)
    d=parse_poses(POSES); Tp=pfirst+d["t_ms"]
    stab={}
    for nm in ("left","right"):
        tr=tracked_arr(d[nm]); d[nm][:,:,:3]=smooth_positions(d[nm],tr); stab[nm]=stability(d[nm],tr)
    Tv=vfirst+video_pts(VIDEO)*1000
    print("detecting hands (MediaPipe)..."); mpc=detect_hands(VIDEO)
    cap=cv2.VideoCapture(VIDEO); fps=cap.get(cv2.CAP_PROP_FPS)
    W=int(cap.get(3)); H=int(cap.get(4))
    vw=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
    last={"left":-999,"right":-999}; n=0
    while True:
        ok,img=cap.read()
        if not ok: break
        if n<len(Tv):
            hp,hr,hands,k=pose_at(d,Tp,Tv[n]+OFFSET)
            cens=mpc[n] if n<len(mpc) else []
            for nm,col in [("left",(0,255,0)),("right",(0,128,255))]:
                if nm not in hands or not(stab[nm][k] and stab[nm][k+1]): continue
                uv,_=cv2.projectPoints(hl(hands[nm],hp,hr).astype(np.float64),RVEC,TVEC,K,DIST)
                uv=uv.reshape(-1,2)
                if present(uv,cens,W,H): last[nm]=n
                elif n-last[nm]>HOLD: continue
                for a,b in BONES:
                    if okp(uv[a]) and okp(uv[b]):
                        cv2.line(img,tuple(uv[a].astype(int)),tuple(uv[b].astype(int)),col,2,cv2.LINE_AA)
                for j in range(26):
                    if okp(uv[j]): cv2.circle(img,tuple(uv[j].astype(int)),3,col,-1,cv2.LINE_AA)
        vw.write(img); n+=1
    cap.release(); vw.release(); print(f"wrote {OUT}: {n} frames")

if __name__=="__main__":
    main()
