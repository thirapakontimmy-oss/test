import cv2, threading, time, random, json, csv, io
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from flask import Flask, Response, jsonify, render_template_string, request, make_response

app = Flask(__name__)

# ── CONFIG ──────────────────────────────────────────────────────
NUM_CAMERAS       = 2
NUM_STUDENTS      = 30
CAM1_SOURCE       = 0       # 0=webcam, 'rtsp://...'=IP Camera
CAM2_SOURCE       = 1
SUMMARY_INTERVAL  = 50*60   # วินาที — สรุปผลทุก 50 นาที (1 คาบ)
CLASS_SUBJECT     = "วิทยาศาสตร์และเทคโนโลยี"
CLASS_ROOM        = "ม.2/1"
TEACHER_NAME      = "ครูผู้สอน"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ── MODELS ──────────────────────────────────────────────────────
@dataclass
class Student:
    student_id: str; name: str; gender: str; class_no: int = 0
    vision_issue: bool = False; hearing_issue: bool = False; adhd_risk: bool = False
    behavior_score: float = 85.0; attention_time: float = 0.0
    distract_count: int = 0; sleep_count: int = 0; session_att: int = 0; session_dis: int = 0
    seat_row: int = 0; seat_col: int = 0; risk_level: str = "low"
    disorder_flags: List[str] = field(default_factory=list)
    advice_tags: List[str] = field(default_factory=list)  # แท็กคำแนะนำ

@dataclass
class ClassSummary:
    summary_id: str; timestamp: str; subject: str; room: str
    duration_min: int; total_students: int
    avg_score: float; avg_attention_pct: float
    top_students: List[dict]; needs_attention: List[dict]
    class_advice: List[str]; behavior_counts: dict
    health_flags: int; disorder_flags: int

NAMES_M = ["สมชาย","วิชัย","ธนากร","ณัฐพล","กิตติพงษ์","ธีรภัทร","วรพล",
           "จักรพล","ภัทรพล","นราธิป","อภิสิทธิ์","ปิยะวัฒน์","สิทธิชัย",
           "ชัยวัฒน์","ธีรวัฒน์","กันตพัฒน์","ปัณณวิชญ์","รัชชานนท์"]
NAMES_F = ["สุภาพร","นภาพร","พรทิพย์","กัญญาณัฐ","ปิยะนุช","ชลธิชา",
           "วันวิสา","ณัฐธิดา","พิมพ์ชนก","อรทัย","ศิริพร","มาลินี",
           "รัตนาภรณ์","วิไลวรรณ","ปรียาภรณ์","กานต์ชนิต","ธัญพิชชา"]
SURNAMES = ["ใจดี","มีสุข","เจริญรุ่ง","ทองดี","สมบูรณ์","ศรีสุข",
            "วงศ์สุวรรณ","อุดมทรัพย์","สว่างไสว","พงษ์ไพร","รักษาดี",
            "ประเสริฐ","เกษมสุข","ชัยมงคล","บุญประเสริฐ"]

def generate_students(n):
    students, used = [], set()
    for i in range(n):
        g = random.choice(['M','F'])
        fn = random.choice(NAMES_M if g=='M' else NAMES_F)
        name = f"{fn} {random.choice(SURNAMES)}"
        att = 0
        while name in used and att < 10:
            fn = random.choice(NAMES_M if g=='M' else NAMES_F)
            name = f"{fn} {random.choice(SURNAMES)}"; att += 1
        used.add(name)
        adhd = random.random() < 0.065
        disorders = []
        if adhd: disorders.append("ADHD")
        if random.random() < 0.03: disorders.append("ASD")
        if random.random() < 0.05: disorders.append("Dyslexia")
        students.append(Student(
            student_id=f"STU{i+1:03d}", name=name, gender=g, class_no=i+1,
            vision_issue=random.random()<0.15, hearing_issue=random.random()<0.08,
            adhd_risk=adhd, behavior_score=random.uniform(55,100), disorder_flags=disorders))
    return students

def assign_seats(students, rows=5, cols=6):
    all_seats=[(r,c) for r in range(rows) for c in range(cols)]
    front=[(r,c) for r in range(2) for c in range(cols)]
    assigned=set()
    random.shuffle(front)
    for s in students:
        if s.vision_issue and front:
            seat=front.pop(0); s.seat_row,s.seat_col=seat; assigned.add(seat)
    for s in students:
        if s.hearing_issue and (s.seat_row,s.seat_col) not in assigned:
            for seat in [(0,1),(0,cols-2),(1,1),(1,cols-2)]:
                if seat not in assigned:
                    s.seat_row,s.seat_col=seat; assigned.add(seat); break
    adhd_cols=set()
    for s in students:
        if s.adhd_risk and (s.seat_row,s.seat_col) not in assigned:
            for seat in all_seats:
                if seat not in assigned and seat[1] not in adhd_cols:
                    s.seat_row,s.seat_col=seat; assigned.add(seat); adhd_cols.add(seat[1]); break
    remaining=[s for s in students if (s.seat_row,s.seat_col)==(0,0)
               and not s.vision_issue and not s.hearing_issue and not s.adhd_risk]
    remaining.sort(key=lambda x:-x.behavior_score)
    avail=[s for s in all_seats if s not in assigned]; random.shuffle(avail)
    for i,s in enumerate(remaining):
        if i<len(avail): s.seat_row,s.seat_col=avail[i]

# ── LEARNING ADVICE ENGINE ────────────────────────────────────────
def generate_advice(s: Student, aff: dict) -> dict:
    """สร้างคำแนะนำการเรียนรายบุคคลจากข้อมูลพฤติกรรมและจิตพิสัย"""
    tips = []
    strategies = []
    parent_msg = []
    strengths = []
    areas = []

    bs = s.behavior_score
    tot = aff.get("total", bs)
    att_min = s.attention_time
    dc = s.distract_count
    sc = s.sleep_count
    flags = s.disorder_flags

    # ── จุดแข็ง ──
    if bs >= 85:
        strengths.append("มีวินัยในตนเองสูง")
    if att_min >= 30:
        strengths.append(f"ตั้งใจเรียนสะสม {att_min:.0f} นาที")
    if aff.get("receiving", 0) >= 80:
        strengths.append("รับรู้และตอบสนองเนื้อหาได้ดี")
    if aff.get("characterization", 0) >= 75:
        strengths.append("มีนิสัยใฝ่เรียนรู้ที่ดี")

    # ── พื้นที่พัฒนา ──
    if dc >= 5:
        areas.append("สมาธิในห้องเรียน")
    if sc >= 2:
        areas.append("ความตื่นตัวและพลังงาน")
    if aff.get("valuing", 0) < 55:
        areas.append("การเห็นคุณค่าในการเรียนรู้")
    if aff.get("organization", 0) < 55:
        areas.append("การจัดระเบียบความคิด")

    # ── กลยุทธ์การเรียนตามพฤติกรรม ──
    if bs >= 80:
        strategies.append("เหมาะสำหรับกิจกรรมผู้นำกลุ่ม — ช่วยสอนเพื่อน")
        strategies.append("ลองท้าทายตัวเองด้วยโจทย์ระดับสูงขึ้น")
        tips.append("ให้นักเรียนเป็น Peer Tutor ในกลุ่มเล็ก")
    elif bs >= 65:
        strategies.append("ใช้วิธีทำซ้ำและทบทวนสม่ำเสมอ")
        strategies.append("จดบันทึกเป็นภาพหรือ Mind Map")
        tips.append("ควรนั่งกลุ่มที่กระตุ้นการมีส่วนร่วม")
    else:
        strategies.append("เริ่มจากเนื้อหาพื้นฐานก่อน ค่อยๆ เพิ่มความยาก")
        strategies.append("ใช้สื่อภาพและวิดีโอช่วยประกอบ")
        tips.append("ครูควรเช็กความเข้าใจเพิ่มเติม 1:1")
        parent_msg.append("นักเรียนอาจต้องการความช่วยเหลือเพิ่มเติมในการทบทวนเนื้อหา")

    if "ADHD" in flags:
        strategies.append("แบ่งงานเป็นชิ้นเล็กๆ มีเป้าหมายชัดเจนในแต่ละขั้น")
        strategies.append("ใช้ Timer Pomodoro: เรียน 15 นาที พัก 5 นาที")
        tips.append("จัดที่นั่งด้านหน้าห้อง ลดสิ่งรบกวน")
        parent_msg.append("แนะนำพบผู้เชี่ยวชาญเพื่อประเมิน ADHD อย่างเป็นทางการ")

    if "Dyslexia" in flags:
        strategies.append("ใช้ข้อความเสียง (Text-to-Speech) ประกอบการอ่าน")
        strategies.append("เขียนด้วยลายมือแทนการพิมพ์ — กระตุ้นความจำ")
        tips.append("ให้เวลาทำข้อสอบเพิ่มตาม IEP")
        parent_msg.append("แนะนำทดสอบการอ่านและเขียนกับนักการศึกษาพิเศษ")

    if sc >= 3:
        strategies.append("ผู้ปกครองควรตรวจสอบเวลานอนหลับ (แนะนำ 8-9 ชม.)")
        parent_msg.append("นักเรียนแสดงสัญญาณอ่อนล้า — ตรวจสอบเวลานอนและโภชนาการ")

    if dc >= 8:
        strategies.append("ลองใช้เทคนิค Active Learning: ถาม-ตอบ กิจกรรม")
        tips.append("จัดกลุ่มกับนักเรียนที่มีสมาธิสูงกว่า")

    # ── ระดับ Krathwohl → คำแนะนำ ──
    lvl = aff.get("krathwohl_level","")
    if "1 —" in lvl:
        strategies.append("กระตุ้นความสนใจด้วยคำถามเปิด และสื่อที่น่าสนใจ")
    elif "2 —" in lvl:
        strategies.append("ส่งเสริมการมีส่วนร่วมในชั้นเรียน — ถามความคิดเห็น")
    elif "3 —" in lvl:
        strategies.append("เชื่อมโยงเนื้อหากับชีวิตจริง — สร้างแรงจูงใจ")
    elif "4 —" in lvl:
        strategies.append("ฝึกคิดวิเคราะห์และสังเคราะห์ข้อมูล")
    elif "5 —" in lvl:
        strategies.append("ท้าทายด้วยโครงงานและการนำเสนอ")

    # Priority tag
    if bs < 60 or sc >= 3 or "ADHD" in flags or "ASD" in flags:
        priority = "high"
        priority_label = "ต้องดูแลพิเศษ"
    elif bs < 75 or dc >= 5:
        priority = "medium"
        priority_label = "ควรติดตาม"
    else:
        priority = "low"
        priority_label = "ปกติ"

    return {
        "student_id":   s.student_id,
        "name":         s.name,
        "gender":       s.gender,
        "behavior_score": round(bs, 1),
        "total_score":  round(tot, 1),
        "krathwohl":    lvl,
        "attention_min":round(att_min, 1),
        "distract_count": dc,
        "sleep_count":  sc,
        "disorder_flags": flags,
        "strengths":    strengths[:3],
        "areas":        areas[:3],
        "strategies":   strategies[:4],
        "teacher_tips": tips[:3],
        "parent_msg":   parent_msg[:2],
        "priority":     priority,
        "priority_label": priority_label,
    }

# ── CLASS SUMMARY ENGINE ──────────────────────────────────────────
def generate_class_summary(elapsed_min: int) -> dict:
    """สรุปผลรายคาบ — เรียกได้ทุกเมื่อหรืออัตโนมัติท้ายคาบ"""
    scores = list(aff_scores.values())
    if not scores:
        return {"error": "ยังไม่มีข้อมูลเพียงพอ"}

    avg_score = round(sum(s.get("total",0) for s in scores) / len(scores), 1)
    avg_att   = round(sum(s.get("attention_min",0) for s in scores) / len(scores), 1)

    # จัดลำดับ
    sorted_s = sorted(scores, key=lambda x: -x.get("total",0))
    top5  = [{"student_id":s["student_id"],"name":s["name"],"total":s.get("total",0),
              "level":s.get("krathwohl_level","")} for s in sorted_s[:5]]
    need5 = [{"student_id":s["student_id"],"name":s["name"],"total":s.get("total",0),
              "risk":s.get("risk_level",""),"flags":s.get("disorder_flags",[])}
             for s in sorted_s[-5:] if s.get("total",0) < 65]

    # คำแนะนำระดับห้องเรียน
    class_advice = []
    low_att = sum(1 for s in students if s.session_att < s.session_dis)
    if low_att > NUM_STUDENTS * 0.3:
        class_advice.append("นักเรียนมากกว่า 30% มีสมาธิต่ำ — แนะนำสลับกิจกรรมทุก 15 นาที")
    if avg_score < 60:
        class_advice.append("คะแนนจิตพิสัยเฉลี่ยต่ำ — ทบทวนเนื้อหาหลักและตรวจสอบความเข้าใจ")
    elif avg_score >= 80:
        class_advice.append("ห้องเรียนมีส่วนร่วมดีเยี่ยม — เหมาะสำหรับกิจกรรมขยายผล")
    sleepers = sum(1 for s in students if s.sleep_count >= 2)
    if sleepers >= 3:
        class_advice.append(f"พบนักเรียนหลับ {sleepers} คน — ตรวจสอบอุณหภูมิห้อง/แสงสว่าง และสอบถามผู้ปกครอง")
    disorder_c = sum(1 for s in students if s.disorder_flags)
    if disorder_c > 0:
        class_advice.append(f"มีนักเรียน {disorder_c} คนที่มีความเสี่ยงพฤติกรรมพิเศษ — แนะนำแจ้งผู้ปกครองเพื่อตรวจเพิ่มเติม")
    if not class_advice:
        class_advice.append("ภาพรวมชั้นเรียนเป็นไปด้วยดี — รักษาระดับการสอนและกิจกรรมต่อไป")

    behavior_counts = {"attentive":0,"distracted":0,"sleeping":0,"sick":0,"aggressive":0}
    for e in event_log[-200:]:
        if e.get("behavior") in behavior_counts:
            behavior_counts[e["behavior"]] += 1

    sid = f"SUM{datetime.now().strftime('%Y%m%d%H%M')}"
    summary = {
        "summary_id":       sid,
        "timestamp":        datetime.now().strftime("%d/%m/%Y %H:%M"),
        "subject":          CLASS_SUBJECT,
        "room":             CLASS_ROOM,
        "teacher":          TEACHER_NAME,
        "duration_min":     elapsed_min,
        "total_students":   NUM_STUDENTS,
        "avg_score":        avg_score,
        "avg_attention_min":avg_att,
        "top_students":     top5,
        "needs_attention":  need5,
        "class_advice":     class_advice,
        "behavior_counts":  behavior_counts,
        "health_flags":     sum(1 for s in students if s.sleep_count>=2 or s.vision_issue),
        "disorder_flags":   disorder_c,
        "risk_distribution":{
            "low":   sum(1 for s in students if s.risk_level=="low"),
            "medium":sum(1 for s in students if s.risk_level=="medium"),
            "high":  sum(1 for s in students if s.risk_level=="high"),
        }
    }
    summaries.append(summary)
    if len(summaries) > 20: summaries[:] = summaries[-20:]
    return summary

# ── CAMERA ──────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, cam_id, source):
        self.cam_id=cam_id; self.source=source; self.label=f"CAM-{cam_id:02d}"
        self.cap=None; self.frame=None; self.annotated=None; self.detections=[]
        self.connected=False; self.running=False; self.fps=0.0
        self.lock=threading.Lock(); self._fc=self._fps_fc=0; self._fps_t=time.time()

    def start(self):
        try:
            self.cap=cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print(f"  ✗ {self.label}: ไม่พบกล้อง"); return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            self.cap.set(cv2.CAP_PROP_FPS,20); self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            self.connected=self.running=True
            threading.Thread(target=self._loop,daemon=True).start()
            print(f"  ✓ {self.label}: เชื่อมต่อสำเร็จ"); return True
        except Exception as e:
            print(f"  ✗ {self.label}: {e}"); return False

    def _loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened(): time.sleep(0.1); continue
            ret,frame=self.cap.read()
            if not ret: time.sleep(0.05); continue
            self._fc+=1; self._fps_fc+=1
            t=time.time()
            if t-self._fps_t>=2.0:
                self.fps=self._fps_fc/(t-self._fps_t); self._fps_fc=0; self._fps_t=t
            if self._fc%4==0:
                ann,dets=self._analyze(frame)
                with self.lock: self.frame=frame; self.annotated=ann; self.detections=dets
            else:
                with self.lock:
                    self.frame=frame
                    if self.annotated is None: self.annotated=frame
            time.sleep(0.03)

    def _analyze(self, frame):
        out=frame.copy()
        gray=cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        dets=[]
        faces=face_cascade.detectMultiScale(gray,1.1,5,minSize=(40,40),flags=cv2.CASCADE_SCALE_IMAGE)
        for i,(x,y,w,h) in enumerate(faces):
            fg=gray[y:y+h,x:x+w]
            eyes=eye_cascade.detectMultiScale(fg,1.1,3,minSize=(15,15))
            beh,conf,lbl=self._classify(fg,eyes)
            c={"attentive":(46,160,90),"distracted":(37,150,190),
               "sleeping":(130,100,80),"sick":(60,80,200),"aggressive":(40,40,200)}.get(beh,(150,150,150))
            cv2.rectangle(out,(x,y),(x+w,y+h),c,2)
            tag=f"{lbl} {conf*100:.0f}%"
            (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
            cv2.rectangle(out,(x,y-th-8),(x+tw+6,y),c,-1)
            cv2.putText(out,tag,(x+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1,cv2.LINE_AA)
            for ex,ey,ew,eh in eyes[:2]: cv2.circle(out,(x+ex+ew//2,y+ey+eh//2),3,(255,220,50),-1)
            dets.append({"timestamp":datetime.now().strftime("%H:%M:%S"),
                         "camera_id":self.cam_id,"face_id":i+1,
                         "behavior":beh,"label":lbl,"confidence":round(conf,2),"eyes":int(len(eyes))})
        self._hud(out,len(faces)); return out,dets

    def _classify(self,fg,eyes):
        n=len(eyes)
        if n>=2:
            area=fg.shape[0]*fg.shape[1]; ea=sum(ew*eh for _,_,ew,eh in eyes[:2])/2
            if area and ea/area<0.015: return "sleeping",0.80,"ตาหรี่/ง่วงนอน"
            return "attentive",0.90,"ตั้งใจเรียน"
        elif n==1: return "distracted",0.73,"วอกแวก/มองข้าง"
        else:
            if float(np.mean(fg))<55: return "sleeping",0.78,"ก้มหน้า/หลับ"
            return "distracted",0.65,"หันหน้าออก"

    def _hud(self,frame,nf):
        h,w=frame.shape[:2]; ov=frame.copy()
        cv2.rectangle(ov,(0,0),(w,32),(20,35,60),-1); cv2.addWeighted(ov,0.7,frame,0.3,0,frame)
        ts=datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        cv2.putText(frame,f"REC  {ts}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.46,(100,200,120),1)
        cv2.putText(frame,f"{self.label}  {self.fps:.1f}fps  ใบหน้า:{nf}",(w//2-120,20),cv2.FONT_HERSHEY_SIMPLEX,0.44,(160,200,240),1)
        if int(time.time()*2)%2==0: cv2.circle(frame,(w-16,16),5,(60,60,220),-1)
        cv2.rectangle(frame,(0,h-22),(w,h),(20,35,60),-1)
        cv2.putText(frame,"ระบบวิเคราะห์ห้องเรียนอัจฉริยะ | กระทรวงศึกษาธิการ",(8,h-6),cv2.FONT_HERSHEY_SIMPLEX,0.34,(100,120,150),1)

    def stream_generator(self):
        while True:
            with self.lock: f=self.annotated if self.annotated is not None else self.frame
            if f is not None:
                _,buf=cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,78])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n'
            time.sleep(0.04)

    def get_detections(self):
        with self.lock: return list(self.detections)
    def stop(self):
        self.running=False
        if self.cap: self.cap.release()

# ── GLOBAL STATE ─────────────────────────────────────────────────
students     = generate_students(NUM_STUDENTS)
assign_seats(students)
cameras: Dict[int,CameraStream] = {}
event_log:   List[dict] = []
aff_scores:  Dict[str,dict] = {}
alert_log:   List[dict] = []
summaries:   List[dict] = []
session_start = datetime.now()
last_summary_time = datetime.now()

def init_cameras():
    for i,src in enumerate([CAM1_SOURCE,CAM2_SOURCE],1):
        if i<=NUM_CAMERAS:
            cam=CameraStream(i,src); cam.start(); cameras[i]=cam

def _update_risk(s):
    s.risk_level="low" if s.behavior_score>=75 else "medium" if s.behavior_score>=50 else "high"

def background_loop():
    global last_summary_time
    while True:
        all_dets=[]
        for cam in cameras.values():
            if cam.connected: all_dets.extend(cam.get_detections())
        now_str=datetime.now().strftime("%H:%M:%S")
        for det in all_dets:
            s=random.choice(students); b=det["behavior"]
            if b=="attentive":
                s.behavior_score=min(100,s.behavior_score+0.4)
                s.attention_time=round(s.attention_time+2/60,2)
                s.session_att+=1
            elif b=="sleeping":
                s.behavior_score=max(0,s.behavior_score-1.8)
                s.sleep_count+=1; s.session_dis+=1
            elif b=="distracted":
                s.behavior_score=max(0,s.behavior_score-0.6)
                s.distract_count+=1; s.session_dis+=1
            elif b=="aggressive":
                s.behavior_score=max(0,s.behavior_score-3.0)
                alert_log.append({"time":now_str,"type":"aggressive","severity":"high",
                    "msg":f"ตรวจพบพฤติกรรมก้าวร้าว | กล้อง CAM-{det['camera_id']:02d}",
                    "student_id":s.student_id,"student_name":s.name})
            _update_risk(s)
            if s.distract_count>10 and not s.adhd_risk:
                s.adhd_risk=True
                if "ADHD" not in s.disorder_flags: s.disorder_flags.append("ADHD")
            event_log.append({**det,"student_id":s.student_id,"student_name":s.name})

        if len(event_log)>500: event_log[:]=event_log[-500:]
        if len(alert_log)>100: alert_log[:]=alert_log[-100:]

        for s in students:
            rec=[e for e in event_log[-100:] if e.get("student_id")==s.student_id]
            att_r=sum(1 for e in rec if e["behavior"]=="attentive")/max(len(rec),1)
            bs=s.behavior_score
            rv  =round(min(100,att_r*100+random.uniform(-3,3)),1)
            rs  =round(min(100,bs*0.90+random.uniform(-4,4)),1)
            val =round(min(100,bs*0.85+random.uniform(-4,4)),1)
            org =round(min(100,bs*0.80+random.uniform(-4,4)),1)
            char=round(min(100,bs*0.75+random.uniform(-4,4)),1)
            tot =round((rv+rs+val+org+char)/5,1)
            lvl=("5 — Characterization" if tot>=80 else "4 — Organization" if tot>=65
                 else "3 — Valuing" if tot>=50 else "2 — Responding" if tot>=35 else "1 — Receiving")
            aff_scores[s.student_id]={
                "student_id":s.student_id,"name":s.name,"class_no":s.class_no,
                "receiving":max(0,rv),"responding":max(0,rs),"valuing":max(0,val),
                "organization":max(0,org),"characterization":max(0,char),"total":max(0,tot),
                "krathwohl_level":lvl,"risk_level":s.risk_level,
                "attention_min":s.attention_time,"distract_count":s.distract_count,
                "disorder_flags":s.disorder_flags,"seat_row":s.seat_row,"seat_col":s.seat_col}

        # Auto summary every SUMMARY_INTERVAL seconds
        elapsed=(datetime.now()-last_summary_time).seconds
        if elapsed>=SUMMARY_INTERVAL:
            elapsed_total=(datetime.now()-session_start).seconds//60
            generate_class_summary(elapsed_total)
            last_summary_time=datetime.now()
            print(f"  ✓ Auto-summary generated at {now_str}")

        time.sleep(2)

# ── ROUTES ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML_PAGE,
        subject=CLASS_SUBJECT, room=CLASS_ROOM, teacher=TEACHER_NAME,
        num_cameras=NUM_CAMERAS, num_students=NUM_STUDENTS,
        session_start=session_start.strftime("%d/%m/%Y %H:%M"))

@app.route('/stream/<int:cam_id>')
def stream(cam_id):
    cam=cameras.get(cam_id)
    if cam and cam.connected:
        return Response(cam.stream_generator(),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(_offline_gen(cam_id),mimetype='multipart/x-mixed-replace; boundary=frame')

def _offline_gen(cam_id):
    img=np.full((480,640,3),28,dtype=np.uint8)
    cv2.putText(img,f"CAM-{cam_id:02d}  ไม่พบสัญญาณกล้อง",(80,230),cv2.FONT_HERSHEY_SIMPLEX,0.9,(70,80,100),2)
    _,buf=cv2.imencode('.jpg',img); fb=buf.tobytes()
    while True: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+fb+b'\r\n'; time.sleep(1)

@app.route('/api/status')
def api_status():
    cam_info={str(cid):{"connected":c.connected,"fps":round(c.fps,1),"label":c.label} for cid,c in cameras.items()}
    all_dets=[]
    for c in cameras.values():
        if c.connected: all_dets.extend(c.get_detections())
    counts={k:0 for k in ["attentive","distracted","sleeping","sick","aggressive"]}
    for d in all_dets:
        if d.get("behavior") in counts: counts[d["behavior"]]+=1
    safety="safe"; alerts=[]
    ra=[a for a in alert_log if a.get("type")=="aggressive"][-5:]
    if ra: safety="danger"; alerts.append(f"ตรวจพบพฤติกรรมก้าวร้าว {len(ra)} ครั้ง — แจ้งครูประจำชั้น")
    if counts["sleeping"]>=3: safety="warning" if safety=="safe" else safety; alerts.append(f"พบนักเรียนหลับ {counts['sleeping']} คน")
    if counts["sick"]>=2: alerts.append(f"นักเรียน {counts['sick']} คนอาจป่วย — ส่งพบพยาบาล")
    hr=[s for s in students if s.risk_level=="high"]
    if len(hr)>=3: alerts.append(f"นักเรียนกลุ่มเสี่ยงสูง {len(hr)} คน — แนะนำแจ้งผู้ปกครอง")
    avg=sum(s.behavior_score for s in students)/len(students)
    elapsed=(datetime.now()-session_start).seconds//60
    next_sum=max(0,SUMMARY_INTERVAL-((datetime.now()-last_summary_time).seconds))
    return jsonify({"cameras":cam_info,"behavior":counts,"total_faces":sum(counts.values()),
        "avg_score":round(avg,1),"safety_status":safety,"alerts":alerts,
        "total_students":len(students),"session_min":elapsed,
        "high_risk_count":len(hr),"adhd_count":sum(1 for s in students if s.adhd_risk),
        "next_summary_sec":next_sum,"summaries_count":len(summaries)})

@app.route('/api/events')
def api_events(): return jsonify(event_log[-40:][::-1])

@app.route('/api/alerts')
def api_alerts(): return jsonify(alert_log[-20:][::-1])

@app.route('/api/affective')
def api_affective():
    data=sorted(aff_scores.values(),key=lambda x:-x["total"]); return jsonify(data)

@app.route('/api/health')
def api_health():
    risks=[]
    rec100=event_log[-100:]
    DISORDERS={
        "ADHD":{"icon":"activity","label":"สมาธิสั้น (ADHD)","desc":"พฤติกรรมวอกแวกบ่อย ไม่นิ่ง สอดคล้อง DSM-5",
                "ref":"ราชวิทยาลัยกุมารแพทย์ฯ: พบในเด็กไทย 5-8%","rec":"แนะนำผู้ปกครองพบจิตแพทย์เด็ก"},
        "ASD":{"icon":"brain","label":"ออทิสติก (ASD)","desc":"รูปแบบพฤติกรรมซ้ำ ไม่ตอบสนองสังคม",
               "ref":"อ้างอิง: ViTASD Facial Expression Model (IrohXu)","rec":"ส่งตรวจประเมินพัฒนาการ"},
        "Dyslexia":{"icon":"book","label":"บกพร่องการอ่าน (Dyslexia)","desc":"มีปัญหาการอ่าน-เขียน",
                    "ref":"อ้างอิง: Behavioural Disorder Detection (Sheetal et al.)","rec":"ส่งต่อนักการศึกษาพิเศษ"},
    }
    HEALTH_RISKS={
        "fatigue":{"icon":"moon","label":"ความเหนื่อยล้า","symptoms":["หาวบ่อย","วางหัวบนโต๊ะ"],"rec":"อาจนอนไม่พอ แจ้งผู้ปกครอง"},
        "eye_strain":{"icon":"eye","label":"สายตาอ่อนล้า","symptoms":["ถูตาบ่อย","มองใกล้"],"rec":"นั่งห่างกระดาน 2 เมตร พักสายตาทุก 20 นาที"},
        "stress":{"icon":"alert-triangle","label":"ความเครียดสะสม","symptoms":["กัดเล็บ","นั่งบิดตัว"],"rec":"พูดคุยกับครูแนะแนว"},
    }
    for s in students:
        for flag in s.disorder_flags:
            if flag in DISORDERS:
                info=DISORDERS[flag]
                risks.append({"student_id":s.student_id,"name":s.name,"category":"disorder",
                    "icon":info["icon"],"label":info["label"],"probability":round(random.uniform(55,85),1),
                    "desc":info["desc"],"ref":info["ref"],"symptoms":[],"recommendation":info["rec"]})
        evs=[e for e in rec100 if e.get("student_id")==s.student_id]
        sleep_n=sum(1 for e in evs if e["behavior"]=="sleeping")
        dist_n=sum(1 for e in evs if e["behavior"]=="distracted")
        if sleep_n>=2:
            ri=HEALTH_RISKS["fatigue"]
            risks.append({"student_id":s.student_id,"name":s.name,"category":"health",
                "icon":ri["icon"],"label":ri["label"],"probability":round(min(95,40+sleep_n*15),1),
                "symptoms":ri["symptoms"],"recommendation":ri["rec"],"desc":"","ref":""})
        if s.vision_issue:
            ri=HEALTH_RISKS["eye_strain"]
            risks.append({"student_id":s.student_id,"name":s.name,"category":"health",
                "icon":ri["icon"],"label":ri["label"],"probability":round(random.uniform(62,88),1),
                "symptoms":ri["symptoms"],"recommendation":ri["rec"],"desc":"","ref":""})
        if dist_n>=5 and s.adhd_risk:
            ri=HEALTH_RISKS["stress"]
            risks.append({"student_id":s.student_id,"name":s.name,"category":"health",
                "icon":ri["icon"],"label":ri["label"],"probability":round(min(90,35+dist_n*5),1),
                "symptoms":ri["symptoms"],"recommendation":ri["rec"],"desc":"","ref":""})
    return jsonify(risks[:25])

@app.route('/api/seating')
def api_seating():
    return jsonify([{"student_id":s.student_id,"name":s.name,"class_no":s.class_no,
        "seat_row":s.seat_row,"seat_col":s.seat_col,"vision_issue":s.vision_issue,
        "hearing_issue":s.hearing_issue,"adhd_risk":s.adhd_risk,"disorder_flags":s.disorder_flags,
        "behavior_score":round(s.behavior_score,1),"risk_level":s.risk_level,
        "attention_min":s.attention_time,"distract_count":s.distract_count} for s in students])

@app.route('/api/advice')
def api_advice():
    """คำแนะนำการเรียนรายบุคคล"""
    result=[]
    for s in students:
        aff=aff_scores.get(s.student_id,{})
        result.append(generate_advice(s,aff))
    result.sort(key=lambda x:{"high":0,"medium":1,"low":2}.get(x["priority"],3))
    return jsonify(result)

@app.route('/api/advice/<student_id>')
def api_advice_one(student_id):
    s=next((x for x in students if x.student_id==student_id),None)
    if not s: return jsonify({"error":"ไม่พบนักเรียน"}),404
    aff=aff_scores.get(student_id,{})
    return jsonify(generate_advice(s,aff))

@app.route('/api/summary', methods=['GET'])
def api_summary():
    """รายการสรุปผลทั้งหมด"""
    return jsonify(summaries[::-1])

@app.route('/api/summary/now', methods=['POST'])
def api_summary_now():
    """สรุปผลทันที"""
    elapsed=(datetime.now()-session_start).seconds//60
    s=generate_class_summary(elapsed)
    return jsonify(s)

@app.route('/api/summary/latest')
def api_summary_latest():
    if summaries: return jsonify(summaries[-1])
    elapsed=(datetime.now()-session_start).seconds//60
    return jsonify(generate_class_summary(elapsed))

@app.route('/api/export/csv')
def export_csv():
    """Export รายงานนักเรียนทั้งหมดเป็น CSV"""
    output=io.StringIO()
    w=csv.writer(output)
    w.writerow(["รหัส","ชื่อ-สกุล","คะแนนพฤติกรรม","คะแนนจิตพิสัย","ระดับ Krathwohl",
                "เวลาตั้งใจ(นาที)","วอกแวก","หลับ","ความเสี่ยง","ความผิดปกติ","คำแนะนำ"])
    for s in students:
        aff=aff_scores.get(s.student_id,{})
        adv=generate_advice(s,aff)
        w.writerow([s.student_id,s.name,round(s.behavior_score,1),
                    aff.get("total",0),aff.get("krathwohl_level",""),
                    round(s.attention_time,1),s.distract_count,s.sleep_count,
                    s.risk_level,",".join(s.disorder_flags),
                    " | ".join(adv["strategies"][:2])])
    resp=make_response(output.getvalue())
    resp.headers["Content-Disposition"]=f"attachment; filename=classroom_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    resp.headers["Content-type"]="text/csv; charset=utf-8-sig"
    return resp


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ระบบวิเคราะห์ห้องเรียนอัจฉริยะ v4</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#f0f4f8;--bg2:#e8edf3;--surf:#fff;--surf2:#f7f9fc;
  --bord:#dde4ee;--bord2:#c8d2e0;
  --navy:#1e3a5f;--blue:#2563b0;--blue2:#3b82d4;--sky:#e8f2fd;
  --teal:#0d9488;--teal2:#e6f7f5;
  --amber:#d97706;--amber2:#fef3c7;
  --red:#dc2626;--red2:#fef2f2;
  --violet:#7c3aed;--violet2:#f3f0ff;
  --green:#16a34a;--green2:#f0fdf4;
  --slate:#64748b;--slate2:#94a3b8;
  --txt:#1e293b;--txt2:#475569;--txt3:#94a3b8;
  --r:10px;--r2:6px;
  --sh:0 1px 3px rgba(0,0,0,.07),0 1px 8px rgba(0,0,0,.04);
  --sh2:0 4px 16px rgba(0,0,0,.10);
  --font:'Noto Sans Thai',sans-serif;--mono:'JetBrains Mono',monospace;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--txt);font-family:var(--font);font-size:15px;line-height:1.6;}
.layout{display:flex;min-height:100vh;}

/* SIDEBAR */
.sidebar{width:232px;flex-shrink:0;background:var(--navy);display:flex;flex-direction:column;
  position:sticky;top:0;height:100vh;overflow-y:auto;}
.slogo{padding:18px 16px 14px;border-bottom:1px solid rgba(255,255,255,.08);}
.slogo-row{display:flex;align-items:center;gap:10px;margin-bottom:4px;}
.slogo-ico{width:34px;height:34px;border-radius:8px;background:linear-gradient(135deg,#3b82f6,#06b6d4);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.slogo-ico svg{width:18px;height:18px;}
.slogo-title{font-size:.9rem;font-weight:700;color:#fff;}
.slogo-sub{font-size:.6rem;color:rgba(255,255,255,.38);font-family:var(--mono);margin-top:3px;line-height:1.6;}
.snav{padding:10px 8px;flex:1;}
.snav-sec{font-size:.6rem;color:rgba(255,255,255,.28);text-transform:uppercase;
  letter-spacing:.1em;padding:13px 8px 5px;}
.ni{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:var(--r2);
  color:rgba(255,255,255,.58);font-size:.85rem;cursor:pointer;transition:.15s;margin-bottom:2px;
  border:none;background:none;width:100%;text-align:left;}
.ni svg{width:16px;height:16px;flex-shrink:0;stroke-width:1.8;}
.ni:hover{background:rgba(255,255,255,.08);color:#fff;}
.ni.on{background:rgba(59,130,212,.4);color:#fff;}
.ni.on svg{color:#7dd3fc;}
.ni .nbadge{margin-left:auto;background:rgba(220,38,38,.7);color:#fff;
  font-size:.6rem;padding:1px 6px;border-radius:10px;font-family:var(--mono);}
.sfoot{padding:12px 16px;border-top:1px solid rgba(255,255,255,.07);
  font-size:.66rem;color:rgba(255,255,255,.26);line-height:1.8;}

/* TOP BAR */
.content{flex:1;display:flex;flex-direction:column;min-width:0;}
.topbar{background:var(--surf);border-bottom:1px solid var(--bord);padding:0 22px;height:52px;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;box-shadow:var(--sh);}
.page-title{font-size:.95rem;font-weight:600;color:var(--navy);}
.tbr{display:flex;align-items:center;gap:11px;}
.live-pill{display:flex;align-items:center;gap:5px;background:var(--teal2);
  border:1px solid #99e6df;border-radius:20px;padding:3px 11px;
  font-size:.7rem;color:var(--teal);font-family:var(--mono);font-weight:500;}
.pulse{width:7px;height:7px;background:var(--teal);border-radius:50%;animation:pls 1.6s infinite;}
@keyframes pls{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
#clk{font-family:var(--mono);font-size:.78rem;color:var(--slate);}
.ses-info{font-size:.7rem;color:var(--slate2);font-family:var(--mono);}
.btn{padding:5px 13px;border-radius:var(--r2);border:1px solid var(--bord2);
  background:var(--surf2);color:var(--txt2);font-family:var(--font);font-size:.78rem;
  cursor:pointer;transition:.15s;display:flex;align-items:center;gap:5px;}
.btn:hover{background:var(--sky);border-color:var(--blue2);color:var(--blue);}
.btn svg{width:14px;height:14px;stroke-width:2;}
.btn-primary{background:var(--blue);color:#fff;border-color:var(--blue);}
.btn-primary:hover{background:var(--blue2);border-color:var(--blue2);color:#fff;}

/* MAIN */
.main{padding:18px 22px 40px;}
.pg{display:none;}.pg.on{display:block;}

/* STAT CARDS */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:15px;}
.card{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);
  padding:14px 16px;box-shadow:var(--sh);position:relative;overflow:hidden;
  transition:box-shadow .2s,transform .2s;animation:ci .3s ease both;}
@keyframes ci{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
.card:hover{box-shadow:var(--sh2);transform:translateY(-1px);}
.cacc{position:absolute;top:0;left:0;right:0;height:3px;border-radius:var(--r) var(--r) 0 0;}
.cico{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;margin-bottom:8px;}
.cico svg{width:16px;height:16px;stroke-width:1.9;}
.clbl{font-size:.68rem;color:var(--slate);font-weight:500;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;}
.cval{font-size:1.8rem;font-weight:700;line-height:1;font-family:var(--mono);}
.csub{font-size:.69rem;color:var(--txt3);margin-top:5px;}
.pwrap{height:4px;background:var(--bord);border-radius:2px;overflow:hidden;margin-top:6px;}
.pfill{height:100%;border-radius:2px;transition:width .8s;}

/* TIMER CARD */
.timer-pill{display:inline-flex;align-items:center;gap:8px;
  background:var(--sky);border:1px solid #bfdbfe;border-radius:var(--r2);
  padding:7px 14px;font-size:.8rem;color:var(--blue);font-family:var(--mono);margin-bottom:15px;}
.timer-pill svg{width:15px;height:15px;}

/* PANELS */
.row2{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-bottom:13px;}
@media(max-width:860px){.row2{grid-template-columns:1fr;}}
.panel{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);box-shadow:var(--sh);overflow:hidden;}
.ph{padding:11px 15px;border-bottom:1px solid var(--bord);display:flex;align-items:center;
  justify-content:space-between;background:var(--surf2);}
.phl{display:flex;align-items:center;gap:8px;}
.phico{width:26px;height:26px;border-radius:6px;display:flex;align-items:center;justify-content:center;}
.phico svg{width:14px;height:14px;stroke-width:2;}
.ptitle{font-size:.8rem;font-weight:600;color:var(--navy);}
.psubt{font-size:.68rem;color:var(--slate2);}
.pb{padding:13px 15px;}

/* ALERTS */
.awrap{margin-bottom:13px;}
.aitem{display:flex;align-items:flex-start;gap:9px;padding:9px 13px;border-radius:var(--r2);
  margin-bottom:5px;font-size:.81rem;border-left:3px solid transparent;}
.aitem svg{width:14px;height:14px;flex-shrink:0;margin-top:2px;}
.aitem.danger{background:var(--red2);border-color:var(--red);color:#991b1b;}
.aitem.warn{background:var(--amber2);border-color:var(--amber);color:#92400e;}
.aitem.info{background:var(--sky);border-color:var(--blue);color:#1e40af;}

/* FEED */
.feed{max-height:260px;overflow-y:auto;}
.fitem{display:flex;align-items:center;gap:7px;padding:6px 0;
  border-bottom:1px solid var(--bord);font-size:.79rem;animation:fi .25s;}
@keyframes fi{from{opacity:0;transform:translateX(-5px)}to{opacity:1}}
.fitem:last-child{border:none;}
.fbadge{padding:2px 7px;border-radius:4px;font-size:.65rem;font-family:var(--mono);font-weight:600;white-space:nowrap;}
.b-att{background:#dcfce7;color:#15803d;}.b-dis{background:#fff7ed;color:#c2410c;}
.b-slp{background:#f1f5f9;color:#64748b;}.b-sic{background:#fef2f2;color:#b91c1c;}
.b-agg{background:#fdf4ff;color:#7e22ce;}
.fdesc{color:var(--txt2);flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.ftime{font-family:var(--mono);font-size:.65rem;color:var(--txt3);white-space:nowrap;}

/* CAMERAS */
.cam-grid{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-bottom:13px;}
@media(max-width:680px){.cam-grid{grid-template-columns:1fr;}}
.cam-wrap{background:#141e30;border:1px solid var(--bord2);border-radius:var(--r);
  overflow:hidden;position:relative;aspect-ratio:4/3;box-shadow:var(--sh);}
.cam-wrap img{width:100%;height:100%;object-fit:cover;display:block;}
.cam-badge{position:absolute;top:9px;left:9px;background:rgba(20,30,50,.82);
  border:1px solid rgba(59,130,180,.45);border-radius:5px;padding:2px 9px;
  font-size:.66rem;font-family:var(--mono);color:#93c5fd;backdrop-filter:blur(4px);}
.cam-rec{position:absolute;top:9px;right:9px;display:flex;align-items:center;gap:4px;
  font-size:.66rem;color:#6ee7b7;font-family:var(--mono);}
.cam-bar{position:absolute;bottom:0;left:0;right:0;background:rgba(12,18,30,.78);
  padding:5px 10px;display:flex;justify-content:space-between;
  font-size:.64rem;font-family:var(--mono);color:rgba(180,210,255,.65);}

/* TABLE */
.tbl{width:100%;border-collapse:collapse;font-size:.79rem;}
.tbl th{padding:8px 11px;text-align:left;font-size:.66rem;text-transform:uppercase;
  letter-spacing:.06em;color:var(--slate);border-bottom:2px solid var(--bord);
  background:var(--surf2);white-space:nowrap;}
.tbl td{padding:7px 11px;border-bottom:1px solid var(--bord);vertical-align:middle;}
.tbl tr:hover td{background:#f5f8fd;}
.sbar{display:flex;align-items:center;gap:7px;}
.strk{flex:1;height:5px;background:var(--bord);border-radius:3px;overflow:hidden;min-width:45px;}
.sfill{height:100%;border-radius:3px;transition:width .5s;}
.fg{background:var(--teal);}.fm{background:var(--amber);}.fb{background:var(--red);}
.snum{font-family:var(--mono);font-size:.7rem;min-width:26px;color:var(--txt2);}
.rbadge{display:inline-flex;align-items:center;gap:3px;padding:2px 7px;border-radius:12px;font-size:.68rem;font-weight:600;}
.r-low{background:#dcfce7;color:#15803d;}.r-medium{background:#fff7ed;color:#c2410c;}.r-high{background:#fef2f2;color:#991b1b;}
.dtag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:.62rem;font-weight:600;margin-right:2px;
  background:var(--violet2);color:var(--violet);border:1px solid #c4b5fd;}
.ptag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:.62rem;font-weight:600;
  background:var(--sky);color:var(--blue);border:1px solid #93c5fd;}

/* HEALTH CARDS */
.hcard{background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);
  padding:12px;margin-bottom:8px;display:flex;gap:10px;align-items:flex-start;transition:box-shadow .2s;}
.hcard:hover{box-shadow:var(--sh2);}
.hcico{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.hcico svg{width:18px;height:18px;stroke-width:1.8;}
.hcbody{flex:1;}
.hctitle{font-weight:600;font-size:.85rem;color:var(--navy);}
.hcstu{font-size:.68rem;color:var(--slate2);font-family:var(--mono);}
.hcprob{font-size:.68rem;font-family:var(--mono);padding:2px 7px;border-radius:4px;}
.hcdesc{font-size:.73rem;color:var(--txt2);margin-top:4px;}
.hcsym{font-size:.71rem;color:var(--slate);margin-top:3px;}
.hcrec{font-size:.71rem;color:var(--teal);margin-top:4px;font-weight:500;}
.hcref{font-size:.64rem;color:var(--slate2);margin-top:2px;font-style:italic;}
.hc-h{background:var(--amber2);border-color:#fde68a;}
.hc-h .hcico{background:#fef3c7;color:var(--amber);}
.hc-h .hcprob{background:var(--amber2);color:var(--amber);}
.hc-d{background:var(--violet2);border-color:#c4b5fd;}
.hc-d .hcico{background:#ede9fe;color:var(--violet);}
.hc-d .hcprob{background:#ede9fe;color:var(--violet);}

/* SAFETY */
.sw{border-radius:var(--r2);padding:12px 15px;text-align:center;margin-bottom:10px;}
.sw.safe{background:var(--green2);border:1px solid #86efac;}
.sw.warning{background:var(--amber2);border:1px solid #fcd34d;}
.sw.danger{background:var(--red2);border:1px solid #fca5a5;}
.sw-ico{font-size:1.6rem;margin-bottom:2px;}
.sw-txt{font-weight:700;font-size:.94rem;}
.sw-sub{font-size:.7rem;color:var(--txt2);margin-top:2px;}
.sm{display:flex;gap:7px;flex-wrap:wrap;}
.smi{flex:1;min-width:82px;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:6px 10px;}
.smv{font-size:1.15rem;font-weight:700;font-family:var(--mono);}
.sml{font-size:.64rem;color:var(--slate);}

/* SEATING */
.sc{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);padding:18px;margin-bottom:13px;}
.blbl{text-align:center;background:var(--navy);color:#93c5fd;border-radius:var(--r2);
  padding:7px;margin-bottom:18px;font-size:.8rem;letter-spacing:.12em;font-weight:600;}
.sg{display:flex;flex-direction:column;align-items:center;gap:5px;}
.sr{display:flex;gap:5px;}
.seat{width:58px;height:49px;border-radius:var(--r2);display:flex;flex-direction:column;
  align-items:center;justify-content:center;cursor:pointer;border:1px solid transparent;transition:all .14s;}
.seat:hover{transform:scale(1.09);box-shadow:var(--sh2);z-index:5;}
.seat.low{background:var(--green2);border-color:#86efac;}
.seat.medium{background:var(--amber2);border-color:#fcd34d;}
.seat.high{background:var(--red2);border-color:#fca5a5;}
.seat.empty{background:var(--bg2);border-color:var(--bord);}
.sico{font-size:.85rem;line-height:1;}
.snam{font-size:.49rem;color:var(--txt2);text-align:center;line-height:1.15;margin-top:1px;
  max-width:52px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.sscr{font-size:.57rem;font-family:var(--mono);font-weight:600;}
.sleg{display:flex;gap:16px;justify-content:center;margin-top:13px;flex-wrap:wrap;font-size:.72rem;color:var(--txt2);}
.li{display:flex;align-items:center;gap:5px;}
.ld{width:11px;height:11px;border-radius:3px;}

/* NOTE */
.cnote{background:var(--sky);border:1px solid #bfdbfe;border-radius:var(--r2);
  padding:7px 13px;font-size:.71rem;color:#1e40af;margin-bottom:9px;
  display:flex;gap:7px;align-items:flex-start;}
.cnote svg{width:13px;height:13px;flex-shrink:0;margin-top:2px;}

/* ─── ADVICE CARDS ─────────────────────────────────────────── */
.adv-filter{display:flex;gap:7px;margin-bottom:14px;flex-wrap:wrap;}
.afbtn{padding:4px 13px;border-radius:20px;border:1px solid var(--bord2);
  background:var(--surf2);color:var(--txt2);font-size:.77rem;cursor:pointer;transition:.15s;}
.afbtn:hover,.afbtn.on{background:var(--blue);color:#fff;border-color:var(--blue);}
.adv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:13px;}
.adv-card{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);
  box-shadow:var(--sh);overflow:hidden;transition:box-shadow .2s;}
.adv-card:hover{box-shadow:var(--sh2);}
.adv-header{padding:12px 14px;display:flex;align-items:flex-start;justify-content:space-between;
  border-bottom:1px solid var(--bord);background:var(--surf2);}
.adv-name{font-weight:600;font-size:.88rem;color:var(--navy);}
.adv-id{font-size:.66rem;color:var(--slate2);font-family:var(--mono);}
.adv-score{font-size:1.4rem;font-weight:700;font-family:var(--mono);}
.adv-body{padding:13px 14px;}
.adv-section{margin-bottom:10px;}
.adv-section:last-child{margin-bottom:0;}
.adv-sec-title{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;
  color:var(--slate);margin-bottom:5px;display:flex;align-items:center;gap:5px;}
.adv-sec-title svg{width:12px;height:12px;}
.adv-tag{display:inline-flex;align-items:center;gap:3px;padding:3px 8px;border-radius:5px;
  font-size:.72rem;margin:2px;line-height:1.4;}
.adv-tag.strength{background:#dcfce7;color:#15803d;}
.adv-tag.area{background:#fff7ed;color:#c2410c;}
.adv-tip{display:flex;align-items:flex-start;gap:7px;padding:5px 0;
  border-bottom:1px solid var(--bord);font-size:.77rem;color:var(--txt2);}
.adv-tip:last-child{border:none;}
.adv-tip-ico{width:16px;height:16px;flex-shrink:0;border-radius:4px;
  display:flex;align-items:center;justify-content:center;font-size:.7rem;margin-top:1px;}
.parent-msg{background:var(--amber2);border:1px solid #fde68a;border-radius:var(--r2);
  padding:8px 10px;font-size:.75rem;color:#92400e;margin-top:6px;}

/* ─── SUMMARY PAGE ─────────────────────────────────────────── */
.sum-hero{background:linear-gradient(135deg,var(--navy),var(--blue));
  color:#fff;border-radius:var(--r);padding:20px 24px;margin-bottom:18px;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;}
.sum-hero-title{font-size:1.1rem;font-weight:700;}
.sum-hero-sub{font-size:.78rem;color:rgba(255,255,255,.65);margin-top:3px;}
.sum-hero-right{display:flex;gap:10px;align-items:center;}
.sum-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:11px;margin-bottom:16px;}
.sum-card{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);
  padding:13px 15px;box-shadow:var(--sh);text-align:center;}
.sum-card-val{font-size:1.7rem;font-weight:700;font-family:var(--mono);}
.sum-card-lbl{font-size:.7rem;color:var(--slate);margin-top:3px;}
.dist-bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:8px 0;}
.dist-seg{height:100%;transition:width .8s;}
.advice-box{background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);
  padding:10px 14px;margin-bottom:8px;display:flex;gap:9px;align-items:flex-start;}
.advice-box svg{width:15px;height:15px;flex-shrink:0;margin-top:2px;color:var(--blue);}
.advice-box.good svg{color:var(--teal);}
.advice-box.warn svg{color:var(--amber);}
.advice-box.alert svg{color:var(--red);}
.sum-list-item{display:flex;align-items:center;gap:8px;padding:6px 0;
  border-bottom:1px solid var(--bord);font-size:.8rem;}
.sum-list-item:last-child{border:none;}
.sum-rank{font-family:var(--mono);font-size:.7rem;color:var(--slate2);width:20px;text-align:right;}
.sum-hist{display:flex;flex-direction:column;gap:8px;max-height:360px;overflow-y:auto;}
.sum-hist-item{background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);
  padding:10px 13px;cursor:pointer;transition:.15s;display:flex;align-items:center;gap:10px;}
.sum-hist-item:hover{background:var(--sky);border-color:var(--blue2);}

/* SCROLLBAR */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:var(--bord2);border-radius:2px;}

/* MODAL */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(15,25,45,.55);
  z-index:500;align-items:center;justify-content:center;backdrop-filter:blur(3px);}
.modal-bg.open{display:flex;}
.modal{background:var(--surf);border-radius:var(--r);box-shadow:0 20px 60px rgba(0,0,0,.2);
  max-width:680px;width:90%;max-height:82vh;overflow:hidden;display:flex;flex-direction:column;}
.modal-head{padding:16px 20px;border-bottom:1px solid var(--bord);
  display:flex;align-items:center;justify-content:space-between;}
.modal-title{font-size:.95rem;font-weight:700;color:var(--navy);}
.modal-close{background:none;border:none;font-size:1.4rem;cursor:pointer;color:var(--slate);line-height:1;}
.modal-body{padding:18px 20px;overflow-y:auto;flex:1;}
</style>
</head>
<body>
<div class="layout">

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="slogo">
    <div class="slogo-row">
      <div class="slogo-ico"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg></div>
      <div><div class="slogo-title">ClassroomAI</div></div>
    </div>
    <div class="slogo-sub">ระบบวิเคราะห์ห้องเรียนอัจฉริยะ v4.0<br>{{ room }} · {{ subject }}</div>
  </div>
  <nav class="snav">
    <div class="snav-sec">หน้าหลัก</div>
    <button class="ni on" onclick="go('overview',this)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>ภาพรวมห้องเรียน</button>
    <button class="ni" onclick="go('cameras',this)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>กล้องวงจรปิด</button>
    <div class="snav-sec">การวิเคราะห์</div>
    <button class="ni" onclick="go('affective',this)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/></svg>คะแนนจิตพิสัย</button>
    <button class="ni" onclick="go('health',this)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>การคัดกรองสุขภาพ</button>
    <button class="ni" onclick="go('safety',this)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>ความปลอดภัย</button>
    <button class="ni" onclick="go('seating',this)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 7H4a2 2 0 00-2 2v6a2 2 0 002 2h16a2 2 0 002-2V9a2 2 0 00-2-2z"/><path d="M16 21V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v16"/></svg>แผนผังที่นั่ง</button>
    <div class="snav-sec">รายงาน</div>
    <button class="ni" onclick="go('advice',this)" id="ni-advice"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>คำแนะนำการเรียน<span class="nbadge" id="adv-badge">!</span></button>
    <button class="ni" onclick="go('summary',this)" id="ni-sum"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>สรุปผลรายคาบ</button>
  </nav>
  <div class="sfoot">เริ่มเซสชัน: {{ session_start }}<br>{{ num_cameras }} กล้อง · {{ num_students }} คน<br>ครู: {{ teacher }}<br>สรุปผลอัตโนมัติทุก 50 นาที</div>
</aside>

<!-- CONTENT -->
<div class="content">
<div class="topbar">
  <div class="page-title" id="page-title">ภาพรวมห้องเรียน</div>
  <div class="tbr">
    <div class="live-pill"><div class="pulse"></div>กำลังวิเคราะห์</div>
    <div class="ses-info" id="ses-info">—</div>
    <div id="clk">--:--:--</div>
    <button class="btn" onclick="exportCSV()"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>Export CSV</button>
    <button class="btn btn-primary" onclick="doSummaryNow()"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>สรุปผลตอนนี้</button>
  </div>
</div>
<div class="main">

<!-- OVERVIEW -->
<div id="pg-overview" class="pg on">
  <div class="timer-pill" id="sum-timer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg><span id="timer-txt">สรุปผลครั้งถัดไป: —</span></div>
  <div class="cards">
    <div class="card"><div class="cacc" style="background:var(--blue)"></div>
      <div class="cico" style="background:var(--sky)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--blue)"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75"/></svg></div>
      <div class="clbl">นักเรียนทั้งหมด</div><div class="cval" id="s-total" style="color:var(--blue)">—</div><div class="csub">ในห้องเรียน</div></div>
    <div class="card"><div class="cacc" style="background:var(--teal)"></div>
      <div class="cico" style="background:var(--teal2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--teal)"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg></div>
      <div class="clbl">ตั้งใจเรียน</div><div class="cval" id="s-att" style="color:var(--teal)">—</div><div class="csub">ตรวจจากกล้อง</div></div>
    <div class="card"><div class="cacc" style="background:var(--amber)"></div>
      <div class="cico" style="background:var(--amber2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--amber)"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
      <div class="clbl">วอกแวก/หลับ</div><div class="cval" id="s-dis" style="color:var(--amber)">—</div><div class="csub">ต้องติดตาม</div></div>
    <div class="card"><div class="cacc" style="background:var(--red)"></div>
      <div class="cico" style="background:var(--red2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--red)"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div>
      <div class="clbl">กลุ่มเสี่ยงสูง</div><div class="cval" id="s-hr" style="color:var(--red)">—</div><div class="csub">ต้องดูแลพิเศษ</div></div>
    <div class="card"><div class="cacc" style="background:var(--violet)"></div>
      <div class="cico" style="background:var(--violet2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--violet)"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
      <div class="clbl">คะแนนเฉลี่ย</div><div class="cval" id="s-avg" style="color:var(--violet)">—</div>
      <div class="csub"><div class="pwrap"><div class="pfill" id="avg-fill" style="width:0%;background:var(--violet)"></div></div></div></div>
    <div class="card"><div class="cacc" style="background:var(--green)"></div>
      <div class="cico" style="background:var(--green2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--green)"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>
      <div class="clbl">รายงานสะสม</div><div class="cval" id="s-sumcnt" style="color:var(--green)">0</div><div class="csub">คาบที่บันทึกแล้ว</div></div>
  </div>
  <div id="awrap" class="awrap"></div>
  <div class="row2">
    <div class="panel">
      <div class="ph"><div class="phl"><div class="phico" style="background:var(--sky)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--blue)"><path d="M13 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V9z"/><polyline points="13 2 13 9 20 9"/></svg></div><span class="ptitle">บันทึกเหตุการณ์</span></div><span class="psubt" id="ev-cnt">—</span></div>
      <div class="pb"><div class="feed" id="feed"></div></div>
    </div>
    <div class="panel">
      <div class="ph"><div class="phl"><div class="phico" style="background:#f3f0ff"><svg viewBox="0 0 24 24" fill="none" stroke="var(--violet)"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div><span class="ptitle">สถานะความปลอดภัย</span></div></div>
      <div class="pb">
        <div class="sw safe" id="sw1"><div class="sw-ico">✅</div><div class="sw-txt">ปลอดภัย</div><div class="sw-sub">ไม่พบเหตุผิดปกติ</div></div>
        <div class="sm" id="cam-sm"></div>
      </div>
    </div>
  </div>
</div>

<!-- CAMERAS -->
<div id="pg-cameras" class="pg">
  <div class="cnote"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>ข้อจำกัด 4.1: ระบุตัวตนโดยอ้างอิงตำแหน่งที่นั่งคงที่ — ผู้เรียนต้องนั่งตามผังที่กำหนด</div>
  <div class="cam-grid">
    <div class="cam-wrap"><img src="/stream/1" alt="CAM-01">
      <div class="cam-badge">CAM-01 | ด้านหน้าห้อง</div>
      <div class="cam-rec"><div class="pulse"></div>กำลังบันทึก</div>
      <div class="cam-bar"><span>Hikvision DS-2CD2143G0-I</span><span id="c1fps">—</span></div>
    </div>
    <div class="cam-wrap"><img src="/stream/2" alt="CAM-02">
      <div class="cam-badge">CAM-02 | ด้านหลังห้อง</div>
      <div class="cam-rec"><div class="pulse"></div>กำลังบันทึก</div>
      <div class="cam-bar"><span>Hikvision DS-2CD2143G0-I</span><span id="c2fps">—</span></div>
    </div>
  </div>
  <div class="panel"><div class="ph"><div class="phl"><div class="phico" style="background:var(--sky)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--blue)"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div><span class="ptitle">Detection Log แบบเรียลไทม์</span></div></div>
  <div class="pb"><div class="feed" id="cam-feed"></div></div></div>
</div>

<!-- AFFECTIVE -->
<div id="pg-affective" class="pg">
  <div class="cnote"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>วัตถุประสงค์ข้อ 1: ลดภาระครู ให้คะแนนทุกคนตาม Krathwohl's Affective Domain อย่างเป็นธรรม</div>
  <div class="panel"><div class="ph"><div class="phl"><div class="phico" style="background:var(--teal2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--teal)"><path d="M9 11l3 3L22 4"/></svg></div><div><div class="ptitle">คะแนนจิตพิสัยรายบุคคล</div><div class="psubt">Receiving · Responding · Valuing · Organization · Characterization</div></div></div><span class="psubt">อัปเดตทุก 2 วินาที</span></div>
  <div style="overflow-x:auto;"><table class="tbl"><thead><tr><th>#</th><th>ชื่อ-สกุล</th><th>การรับรู้</th><th>ตอบสนอง</th><th>ใส่ใจ</th><th>จัดระบบ</th><th>นิสัย</th><th>รวม</th><th>ระดับ Krathwohl</th><th>ความเสี่ยง</th><th>ความผิดปกติ</th></tr></thead><tbody id="aff-body"></tbody></table></div></div>
</div>

<!-- HEALTH -->
<div id="pg-health" class="pg">
  <div class="cnote"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>ข้อจำกัด 4.2: การคัดกรองเบื้องต้นเพื่อสนับสนุนครูและผู้ปกครอง ไม่ใช่การวินิจฉัยทางการแพทย์</div>
  <div class="row2">
    <div><div style="font-size:.75rem;font-weight:600;color:var(--navy);margin-bottom:7px;">ความเสี่ยงด้านสุขภาพ</div><div id="hbox"></div></div>
    <div><div style="font-size:.75rem;font-weight:600;color:var(--navy);margin-bottom:4px;">ความเสี่ยงความผิดปกติพฤติกรรม</div><div style="font-size:.69rem;color:var(--slate2);margin-bottom:7px;">อ้างอิง: ViTASD · ASD-DiagNet · Sheetal et al.</div><div id="dbox"></div></div>
  </div>
</div>

<!-- SAFETY -->
<div id="pg-safety" class="pg">
  <div class="cnote"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>วัตถุประสงค์ข้อ 2: แจ้งเตือนครูภายใน 5 นาที ผ่าน LINE Messaging API</div>
  <div class="row2">
    <div class="panel"><div class="ph"><div class="phl"><div class="phico" style="background:var(--red2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--red)"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div><span class="ptitle">ประวัติการแจ้งเตือน</span></div></div>
    <div class="pb"><div class="feed" id="al-feed"></div></div></div>
    <div class="panel"><div class="ph"><div class="phl"><div class="phico" style="background:#f3f0ff"><svg viewBox="0 0 24 24" fill="none" stroke="var(--violet)"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div><span class="ptitle">สถานะปัจจุบัน</span></div></div>
    <div class="pb">
      <div class="sw safe" id="sw2"><div class="sw-ico">✅</div><div class="sw-txt">ปลอดภัย</div><div class="sw-sub">ไม่พบเหตุผิดปกติ</div></div>
      <div style="font-size:.75rem;font-weight:600;color:var(--navy);margin:10px 0 6px;">ตัวแปรควบคุม (5.3)</div>
      <div style="font-size:.71rem;color:var(--txt2);line-height:1.9;">
        • สภาพแสงสว่างในห้องเรียน<br>• ตำแหน่งและมุมกล้อง (Hikvision DS-2CD2143G0-I)<br>
        • จำนวนผู้เรียน: {{ num_students }} คน<br>• เครือข่าย: Gigabit PoE Switch TP-Link TL-SG1005P
      </div>
    </div></div>
  </div>
  <div class="panel"><div class="ph"><div class="phl"><div class="phico" style="background:var(--sky)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--blue)"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div><span class="ptitle">บันทึกเหตุการณ์ความปลอดภัย</span></div></div>
  <div class="pb"><div class="feed" id="sf-feed"></div></div></div>
</div>

<!-- SEATING -->
<div id="pg-seating" class="pg">
  <div class="cnote"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>ข้อจำกัด 4.1: ผู้เรียนต้องนั่งตามผังที่กำหนด | วัตถุประสงค์ข้อ 4: สภาพแวดล้อมที่เหมาะสมรายบุคคล</div>
  <div class="sc"><div class="blbl">กระดาน / ครูผู้สอน</div><div class="sg" id="seat-map"></div>
    <div class="sleg">
      <div class="li"><div class="ld" style="background:#dcfce7;border:1px solid #86efac"></div>ความเสี่ยงต่ำ</div>
      <div class="li"><div class="ld" style="background:#fff7ed;border:1px solid #fcd34d"></div>ปานกลาง</div>
      <div class="li"><div class="ld" style="background:#fef2f2;border:1px solid #fca5a5"></div>สูง</div>
      <div class="li">👓 สายตาสั้น</div><div class="li">👂 การได้ยิน</div>
      <div class="li" style="color:var(--violet)">⚡ ADHD (กระจายที่นั่ง)</div>
    </div>
  </div>
  <div class="panel"><div class="ph"><div class="phl"><div class="phico" style="background:var(--sky)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--blue)"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/></svg></div><span class="ptitle">รายชื่อและตำแหน่งที่นั่ง</span></div></div>
  <div style="overflow-x:auto;"><table class="tbl"><thead><tr><th>รหัส</th><th>ชื่อ-สกุล</th><th>แถว</th><th>ที่</th><th>สายตา</th><th>การได้ยิน</th><th>ADHD</th><th>คะแนน</th><th>ความเสี่ยง</th><th>ความผิดปกติ</th></tr></thead><tbody id="seat-body"></tbody></table></div></div>
</div>

<!-- ADVICE -->
<div id="pg-advice" class="pg">
  <div class="cnote"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>คำแนะนำรายบุคคลวิเคราะห์จากพฤติกรรม คะแนนจิตพิสัย และประวัติการตั้งใจเรียนในคาบ</div>
  <div class="adv-filter">
    <button class="afbtn on" onclick="filterAdv('all',this)">ทั้งหมด</button>
    <button class="afbtn" onclick="filterAdv('high',this)">ต้องดูแลพิเศษ</button>
    <button class="afbtn" onclick="filterAdv('medium',this)">ควรติดตาม</button>
    <button class="afbtn" onclick="filterAdv('low',this)">ปกติ</button>
  </div>
  <div class="adv-grid" id="adv-grid"></div>
</div>

<!-- SUMMARY -->
<div id="pg-summary" class="pg">
  <div id="sum-content"></div>
  <div class="row2" style="margin-top:16px;">
    <div class="panel">
      <div class="ph"><div class="phl"><div class="phico" style="background:var(--sky)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--blue)"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/></svg></div><span class="ptitle">ประวัติการสรุปผล</span></div><span class="psubt" id="sum-hist-cnt">—</span></div>
      <div class="pb"><div class="sum-hist" id="sum-hist"></div></div>
    </div>
    <div class="panel">
      <div class="ph"><div class="phl"><div class="phico" style="background:var(--green2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--green)"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div><span class="ptitle">สถิติรายคาบล่าสุด</span></div></div>
      <div class="pb" id="sum-stats"></div>
    </div>
  </div>
</div>

</div><!-- /main -->
</div><!-- /content -->
</div><!-- /layout -->

<!-- SUMMARY MODAL -->
<div class="modal-bg" id="sum-modal">
  <div class="modal">
    <div class="modal-head">
      <div class="modal-title" id="modal-title">รายละเอียดการสรุปผล</div>
      <button class="modal-close" onclick="$('sum-modal').classList.remove('open')">×</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const PT={overview:'ภาพรวมห้องเรียน',cameras:'กล้องวงจรปิด',affective:'คะแนนจิตพิสัย',
  health:'การคัดกรองสุขภาพ',safety:'ระบบความปลอดภัย',seating:'แผนผังที่นั่ง',
  advice:'คำแนะนำการเรียนรายบุคคล',summary:'สรุปผลรายคาบ'};
const RC={low:'r-low',medium:'r-medium',high:'r-high'};
const RL={low:'ต่ำ',medium:'ปานกลาง',high:'สูง'};
const BL={attentive:'b-att ✓ ตั้งใจเรียน',distracted:'b-dis วอกแวก',
          sleeping:'b-slp หลับ/ง่วง',sick:'b-sic อาการป่วย',aggressive:'b-agg ก้าวร้าว'};

function sbar(v){v=Math.max(0,v);const c=v>=70?'fg':v>=50?'fm':'fb';
  return `<div class="sbar"><div class="strk"><div class="sfill ${c}" style="width:${v}%"></div></div><span class="snum">${v}</span></div>`;}
function fmtSec(s){const m=Math.floor(s/60),sc=s%60;return `${m}:${String(sc).padStart(2,'0')}`;}

setInterval(()=>{$('clk').textContent=new Date().toLocaleTimeString('th-TH',{hour12:false});},1000);

function go(pg,btn){
  document.querySelectorAll('.pg').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.ni').forEach(b=>b.classList.remove('on'));
  $('pg-'+pg).classList.add('on'); btn.classList.add('on');
  $('page-title').textContent=PT[pg]||pg;
  if(pg==='affective')loadAff();if(pg==='health')loadHealth();
  if(pg==='safety')loadSafety();if(pg==='seating')loadSeating();
  if(pg==='advice')loadAdvice();if(pg==='summary')loadSummary();
}

// ── STATUS ────────────────────────────────────────────────────
async function loadStatus(){
  try{
    const d=await(await fetch('/api/status')).json();
    $('s-total').textContent=d.total_students;
    $('s-att').textContent=d.behavior.attentive;
    $('s-dis').textContent=d.behavior.distracted+d.behavior.sleeping;
    $('s-hr').textContent=d.high_risk_count;
    $('s-avg').textContent=d.avg_score+'%';
    $('avg-fill').style.width=d.avg_score+'%';
    $('s-sumcnt').textContent=d.summaries_count;
    $('ses-info').textContent=`เซสชัน: ${d.session_min} นาที`;

    // Timer
    const ns=d.next_summary_sec;
    const timerEl=$('timer-txt');
    if(ns>0){timerEl.textContent=`สรุปผลครั้งถัดไป: ${fmtSec(ns)}`;}
    else{timerEl.textContent='กำลังสรุปผล...';}
    $('sum-timer').style.background=ns<120?'var(--amber2)':'var(--sky)';
    $('sum-timer').style.borderColor=ns<120?'#fde68a':'#bfdbfe';
    $('sum-timer').style.color=ns<120?'var(--amber)':'var(--blue)';

    // Badges sidebar
    const adv_badge=$('adv-badge');
    if(d.high_risk_count>0){adv_badge.textContent=d.high_risk_count;adv_badge.style.display='';}
    else adv_badge.style.display='none';

    $('awrap').innerHTML=d.alerts.map(a=>{
      const cls=a.includes('ก้าวร้าว')?'danger':a.includes('หลับ')?'warn':'info';
      return `<div class="aitem ${cls}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>${a}</div>`;
    }).join('');

    const si=['safe','warning','danger'];const sico=['✅','⚠️','🚨'];
    const stxt=['ห้องเรียนปลอดภัย','ควรระวัง','อันตราย!'];
    const ssub=['ไม่พบเหตุผิดปกติ','กำลังติดตาม','แจ้งครูทันที'];
    const idx=si.indexOf(d.safety_status);
    for(const id of['sw1','sw2']){const w=$(id);if(!w)continue;
      w.className=`sw ${d.safety_status}`;
      w.querySelector('.sw-ico').textContent=sico[idx];
      w.querySelector('.sw-txt').textContent=stxt[idx];
      w.querySelector('.sw-sub').textContent=ssub[idx];}

    let cm='';
    for(const[id,info] of Object.entries(d.cameras)){
      cm+=`<div class="smi"><div class="smv" style="color:${info.connected?'var(--teal)':'var(--red)'}">${info.connected?'●':'○'}</div><div class="sml">CAM-0${id} ${info.connected?info.fps+'fps':'OFFLINE'}</div></div>`;
      const fe=$('c'+id+'fps');if(fe)fe.textContent=info.connected?info.fps+' fps':'OFFLINE';
    }
    $('cam-sm').innerHTML=cm;
  }catch(e){}
}

async function loadEvents(){
  try{
    const d=await(await fetch('/api/events')).json();
    $('ev-cnt').textContent=d.length+' รายการ';
    renderFeed('feed',d.slice(0,20));renderFeed('cam-feed',d.slice(0,20));
  }catch(e){}
}
function renderFeed(id,evs){
  const el=$(id);if(!el)return;
  el.innerHTML=evs.map(e=>{
    const parts=(BL[e.behavior]||'b-dis ?').split(' ');
    const lbl=parts.slice(1).join(' ')||e.behavior;
    return `<div class="fitem"><span class="fbadge ${parts[0]}">${lbl}</span>
      <span class="fdesc">${e.student_name||''} ${e.label?'— '+e.label:''}</span>
      <span class="ftime">${e.timestamp} CAM-0${e.camera_id}</span></div>`;
  }).join('');}

// ── AFFECTIVE ─────────────────────────────────────────────────
async function loadAff(){
  try{const d=await(await fetch('/api/affective')).json();
    $('aff-body').innerHTML=d.map((s,i)=>{
      const dt=(s.disorder_flags||[]).map(f=>`<span class="dtag">${f}</span>`).join('');
      return `<tr><td style="color:var(--txt3);font-family:var(--mono)">${i+1}</td>
        <td>${s.name}<br><span style="font-size:.63rem;color:var(--txt3);font-family:var(--mono)">${s.student_id}</span></td>
        <td>${sbar(s.receiving)}</td><td>${sbar(s.responding)}</td><td>${sbar(s.valuing)}</td>
        <td>${sbar(s.organization)}</td><td>${sbar(s.characterization)}</td>
        <td><strong style="font-family:var(--mono);color:var(--blue)">${s.total}</strong></td>
        <td style="font-size:.67rem;color:var(--slate)">${s.krathwohl_level}</td>
        <td><span class="rbadge ${RC[s.risk_level]}">${RL[s.risk_level]}</span></td>
        <td>${dt||'—'}</td></tr>`;
    }).join('');}catch(e){}}

// ── HEALTH ────────────────────────────────────────────────────
const HICO={moon:`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>`,
  eye:`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`,
  'alert-triangle':`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
  activity:`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`,
  brain:`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9.5 2a4.5 4.5 0 000 9H12"/><path d="M14.5 2a4.5 4.5 0 010 9H12"/><path d="M12 11v11"/><path d="M7 16.5a4.5 4.5 0 009 0"/></svg>`,
  book:`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>`};

async function loadHealth(){
  try{const d=await(await fetch('/api/health')).json();
    const h=d.filter(r=>r.category==='health');const dr=d.filter(r=>r.category==='disorder');
    const rc=r=>{const cls=r.category==='disorder'?'hc-d':'hc-h';
      return `<div class="hcard ${cls}"><div class="hcico">${HICO[r.icon]||HICO['alert-triangle']}</div><div class="hcbody">
        <div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:3px;">
          <span class="hctitle">${r.label}</span><span class="hcstu">${r.student_id} — ${r.name}</span>
          <span class="hcprob">${r.probability}%</span></div>
        ${r.desc?`<div class="hcdesc">${r.desc}</div>`:''}
        ${r.symptoms?.length?`<div class="hcsym">อาการ: ${r.symptoms.join(', ')}</div>`:''}
        <div class="hcrec">แนะนำ: ${r.recommendation}</div>
        ${r.ref?`<div class="hcref">${r.ref}</div>`:''}
      </div></div>`;};
    $('hbox').innerHTML=h.length?h.map(rc).join(''):'<div style="color:var(--teal);padding:.8rem;font-size:.8rem;">✓ ไม่พบความเสี่ยงด้านสุขภาพ</div>';
    $('dbox').innerHTML=dr.length?dr.map(rc).join(''):'<div style="color:var(--teal);padding:.8rem;font-size:.8rem;">✓ ไม่พบสัญญาณความผิดปกติ</div>';
  }catch(e){}}

// ── SAFETY ────────────────────────────────────────────────────
async function loadSafety(){
  try{
    const al=await(await fetch('/api/alerts')).json();
    const ev=await(await fetch('/api/events')).json();
    renderFeed('sf-feed',ev.filter(e=>e.behavior==='aggressive').slice(0,15));
    $('al-feed').innerHTML=al.length?al.map(a=>`<div class="fitem"><span class="fbadge b-agg">ก้าวร้าว</span><span class="fdesc">${a.msg}</span><span class="ftime">${a.time}</span></div>`).join(''):'<div style="color:var(--teal);font-size:.8rem;padding:.4rem 0;">✓ ไม่มีการแจ้งเตือน</div>';
  }catch(e){}}

// ── SEATING ───────────────────────────────────────────────────
async function loadSeating(){
  try{const d=await(await fetch('/api/seating')).json();
    const g={};let mr=0,mc=0;
    d.forEach(s=>{g[`${s.seat_row}-${s.seat_col}`]=s;mr=Math.max(mr,s.seat_row);mc=Math.max(mc,s.seat_col);});
    let html='';
    for(let r=0;r<=mr;r++){html+='<div class="sr">';
      for(let c=0;c<=mc;c++){const s=g[`${r}-${c}`];
        if(s){const ico=(s.vision_issue?'👓':'')+(s.hearing_issue?'👂':'')+(s.adhd_risk?'⚡':'');
          const sc=s.risk_level==='low'?'var(--teal)':s.risk_level==='medium'?'var(--amber)':'var(--red)';
          html+=`<div class="seat ${s.risk_level}" title="${s.name}\nคะแนน: ${s.behavior_score}"><div class="sico">${ico||'👤'}</div><div class="snam">${s.name.split(' ')[0]}</div><div class="sscr" style="color:${sc}">${s.behavior_score}</div></div>`;}
        else html+=`<div class="seat empty">—</div>`;}
      html+='</div>';}
    $('seat-map').innerHTML=html;
    $('seat-body').innerHTML=d.map(s=>`<tr>
      <td style="font-family:var(--mono);color:var(--txt3);font-size:.7rem">${s.student_id}</td>
      <td>${s.name}</td><td>${s.seat_row+1}</td><td>${s.seat_col+1}</td>
      <td>${s.vision_issue?'<span style="color:var(--blue)">👓 ใช่</span>':'—'}</td>
      <td>${s.hearing_issue?'<span style="color:var(--blue)">👂 ใช่</span>':'—'}</td>
      <td>${s.adhd_risk?'<span style="color:var(--violet)">⚡ เสี่ยง</span>':'—'}</td>
      <td>${sbar(s.behavior_score)}</td>
      <td><span class="rbadge ${RC[s.risk_level]}">${RL[s.risk_level]}</span></td>
      <td>${(s.disorder_flags||[]).map(f=>`<span class="dtag">${f}</span>`).join('')||'—'}</td>
    </tr>`).join('');}catch(e){}}

// ── ADVICE ────────────────────────────────────────────────────
let advData=[];
async function loadAdvice(){
  try{advData=await(await fetch('/api/advice')).json();renderAdvice('all');}catch(e){}
}
function filterAdv(f,btn){
  document.querySelectorAll('.afbtn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');renderAdvice(f);
}
const PRIORITYCLS={high:'var(--red)',medium:'var(--amber)',low:'var(--teal)'};
const PRIORITYBG ={high:'var(--red2)',medium:'var(--amber2)',low:'var(--teal2)'};
function renderAdvice(filter){
  const data=filter==='all'?advData:advData.filter(a=>a.priority===filter);
  $('adv-grid').innerHTML=data.map(a=>{
    const sc=a.total_score>=75?'var(--teal)':a.total_score>=50?'var(--amber)':'var(--red)';
    const dtags=(a.disorder_flags||[]).map(f=>`<span class="dtag">${f}</span>`).join('');
    const strs=(a.strengths||[]).map(s=>`<span class="adv-tag strength">✓ ${s}</span>`).join('');
    const areas=(a.areas||[]).map(s=>`<span class="adv-tag area">→ ${s}</span>`).join('');
    const tips=(a.strategies||[]).map(s=>`<div class="adv-tip"><div class="adv-tip-ico" style="background:var(--sky)">💡</div>${s}</div>`).join('');
    const ttips=(a.teacher_tips||[]).map(s=>`<div class="adv-tip"><div class="adv-tip-ico" style="background:var(--teal2)">👩‍🏫</div>${s}</div>`).join('');
    const pmsg=(a.parent_msg||[]).length?`<div class="parent-msg">📣 สำหรับผู้ปกครอง: ${a.parent_msg.join(' · ')}</div>`:'';
    return `<div class="adv-card" data-priority="${a.priority}">
      <div class="adv-header">
        <div>
          <div class="adv-name">${a.name}</div>
          <div class="adv-id">${a.student_id} · ${a.krathwohl}</div>
          <div style="margin-top:5px;display:flex;gap:5px;flex-wrap:wrap;">
            <span class="rbadge" style="background:${PRIORITYBG[a.priority]};color:${PRIORITYCLS[a.priority]}">${a.priority_label}</span>
            ${dtags}
          </div>
        </div>
        <div style="text-align:right">
          <div class="adv-score" style="color:${sc}">${a.total_score}</div>
          <div style="font-size:.67rem;color:var(--slate2)">คะแนนรวม</div>
          <div style="font-size:.68rem;color:var(--slate);margin-top:4px">สมาธิ ${a.attention_min} นาที</div>
        </div>
      </div>
      <div class="adv-body">
        ${strs||areas?`<div class="adv-section">
          <div class="adv-sec-title"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>จุดแข็ง / พัฒนา</div>
          ${strs}${areas}</div>`:''}
        ${tips?`<div class="adv-section">
          <div class="adv-sec-title"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>กลยุทธ์การเรียน</div>
          ${tips}</div>`:''}
        ${ttips?`<div class="adv-section">
          <div class="adv-sec-title"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/></svg>คำแนะนำสำหรับครู</div>
          ${ttips}</div>`:''}
        ${pmsg}
      </div>
    </div>`;
  }).join('')||'<div style="color:var(--slate2);text-align:center;padding:2rem;font-size:.85rem;">ไม่มีข้อมูลในกลุ่มนี้</div>';
}

// ── SUMMARY ───────────────────────────────────────────────────
async function loadSummary(){
  try{
    const sums=await(await fetch('/api/summary')).json();
    const latest=sums.length?sums[0]:await(await fetch('/api/summary/latest')).json();
    renderSummaryHero(latest);
    $('sum-hist-cnt').textContent=`${sums.length} ครั้ง`;
    $('sum-hist').innerHTML=sums.length?sums.map((s,i)=>`
      <div class="sum-hist-item" onclick="showSumModal(${JSON.stringify(s).replace(/"/g,'&quot;')})">
        <div style="background:var(--sky);color:var(--blue);width:32px;height:32px;border-radius:8px;
          display:flex;align-items:center;justify-content:center;flex-shrink:0;font-weight:700;font-size:.82rem;">${sums.length-i}</div>
        <div style="flex:1">
          <div style="font-size:.82rem;font-weight:600;color:var(--navy)">${s.timestamp}</div>
          <div style="font-size:.7rem;color:var(--slate2)">${s.subject} · ${s.duration_min} นาที · คะแนนเฉลี่ย ${s.avg_score}</div>
        </div>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:14px;height:14px;color:var(--slate2)"><polyline points="9 18 15 12 9 6"/></svg>
      </div>`).join(''):'<div style="color:var(--slate2);font-size:.8rem;text-align:center;padding:1rem;">ยังไม่มีการสรุปผล<br>กด "สรุปผลตอนนี้" เพื่อสร้างรายงาน</div>';

    // Stats panel
    renderSumStats(latest);
  }catch(e){console.error(e);}
}

function renderSummaryHero(s){
  if(!s||s.error){
    $('sum-content').innerHTML='<div style="text-align:center;color:var(--slate2);padding:2rem;font-size:.85rem;">ยังไม่มีการสรุปผล — กด "สรุปผลตอนนี้" ด้านบน</div>';
    return;
  }
  const dist=s.risk_distribution||{low:0,medium:0,high:0};
  const total=dist.low+dist.medium+dist.high||1;
  const adv=(s.class_advice||[]).map(a=>{
    const cls=a.includes('เยี่ยม')||a.includes('ดี')?'good':a.includes('น่าเป็นห่วง')||a.includes('อันตราย')?'alert':'warn';
    return `<div class="advice-box ${cls}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <span>${a}</span></div>`;
  }).join('');
  const tops=(s.top_students||[]).map((t,i)=>`<div class="sum-list-item">
    <span class="sum-rank">${i+1}</span>
    <span style="flex:1;font-weight:500">${t.name}</span>
    <span style="font-family:var(--mono);font-size:.75rem;color:var(--teal)">${t.total}</span>
  </div>`).join('');
  const needs=(s.needs_attention||[]).map(n=>`<div class="sum-list-item">
    <span style="flex:1">${n.name}</span>
    <span class="rbadge r-high" style="font-size:.67rem">${n.total}</span>
    ${(n.flags||[]).map(f=>`<span class="dtag">${f}</span>`).join('')}
  </div>`).join('');

  $('sum-content').innerHTML=`
    <div class="sum-hero">
      <div>
        <div class="sum-hero-title">สรุปผลรายคาบ · ${s.subject}</div>
        <div class="sum-hero-sub">${s.room} · ${s.timestamp} · ระยะเวลา ${s.duration_min} นาที · ครู: ${s.teacher||'—'}</div>
      </div>
      <div class="sum-hero-right">
        <div style="text-align:center"><div style="font-size:1.8rem;font-weight:700;font-family:var(--mono)">${s.avg_score}</div><div style="font-size:.7rem;opacity:.7">คะแนนเฉลี่ย</div></div>
        <div style="text-align:center"><div style="font-size:1.8rem;font-weight:700;font-family:var(--mono)">${s.total_students}</div><div style="font-size:.7rem;opacity:.7">นักเรียน</div></div>
      </div>
    </div>
    <div class="sum-cards">
      <div class="sum-card"><div class="sum-card-val" style="color:var(--green)">${dist.low}</div><div class="sum-card-lbl">ความเสี่ยงต่ำ</div></div>
      <div class="sum-card"><div class="sum-card-val" style="color:var(--amber)">${dist.medium}</div><div class="sum-card-lbl">ความเสี่ยงปานกลาง</div></div>
      <div class="sum-card"><div class="sum-card-val" style="color:var(--red)">${dist.high}</div><div class="sum-card-lbl">ความเสี่ยงสูง</div></div>
      <div class="sum-card"><div class="sum-card-val" style="color:var(--violet)">${s.health_flags||0}</div><div class="sum-card-lbl">สัญญาณสุขภาพ</div></div>
      <div class="sum-card"><div class="sum-card-val" style="color:var(--blue)">${s.disorder_flags||0}</div><div class="sum-card-lbl">ความผิดปกติพฤติกรรม</div></div>
      <div class="sum-card"><div class="sum-card-val" style="color:var(--slate)">${s.avg_attention_min||0}</div><div class="sum-card-lbl">สมาธิเฉลี่ย (นาที)</div></div>
    </div>
    <div style="margin-bottom:14px;">
      <div style="font-size:.75rem;font-weight:600;color:var(--navy);margin-bottom:7px;">การกระจายความเสี่ยงในห้องเรียน</div>
      <div class="dist-bar">
        <div class="dist-seg" style="width:${dist.low/total*100}%;background:var(--teal)"></div>
        <div class="dist-seg" style="width:${dist.medium/total*100}%;background:var(--amber)"></div>
        <div class="dist-seg" style="width:${dist.high/total*100}%;background:var(--red)"></div>
      </div>
      <div style="display:flex;gap:14px;font-size:.7rem;color:var(--slate2);">
        <span style="color:var(--teal)">■ ต่ำ ${Math.round(dist.low/total*100)}%</span>
        <span style="color:var(--amber)">■ กลาง ${Math.round(dist.medium/total*100)}%</span>
        <span style="color:var(--red)">■ สูง ${Math.round(dist.high/total*100)}%</span>
      </div>
    </div>
    <div class="row2">
      <div class="panel"><div class="ph"><div class="phl"><div class="phico" style="background:var(--teal2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--teal)"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg></div><span class="ptitle">Top 5 — ผลงานดีเยี่ยม</span></div></div>
      <div class="pb">${tops||'<div style="color:var(--slate2);font-size:.8rem">ไม่มีข้อมูล</div>'}</div></div>
      <div class="panel"><div class="ph"><div class="phl"><div class="phico" style="background:var(--amber2)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--amber)"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg></div><span class="ptitle">ต้องดูแลเป็นพิเศษ</span></div></div>
      <div class="pb">${needs||'<div style="color:var(--teal);font-size:.8rem">✓ ไม่มีนักเรียนในกลุ่มนี้</div>'}</div></div>
    </div>
    <div class="panel" style="margin-bottom:0">
      <div class="ph"><div class="phl"><div class="phico" style="background:var(--sky)"><svg viewBox="0 0 24 24" fill="none" stroke="var(--blue)"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg></div><span class="ptitle">คำแนะนำสำหรับครูผู้สอน</span></div></div>
      <div class="pb">${adv||'<div style="color:var(--teal);font-size:.8rem">✓ ไม่มีคำแนะนำพิเศษในคาบนี้</div>'}</div>
    </div>`;
}

function renderSumStats(s){
  if(!s||s.error){$('sum-stats').innerHTML='<div style="color:var(--slate2);font-size:.8rem;">รอข้อมูลการสรุปผล...</div>';return;}
  const bc=s.behavior_counts||{};
  $('sum-stats').innerHTML=`
    <div style="margin-bottom:10px;">
      ${Object.entries({attentive:'✓ ตั้งใจเรียน',distracted:'วอกแวก',sleeping:'หลับ',sick:'อาการป่วย',aggressive:'ก้าวร้าว'}).map(([k,l])=>`
        <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--bord);font-size:.79rem;">
          <span style="flex:1;color:var(--txt2)">${l}</span>
          <span style="font-family:var(--mono);font-weight:700;font-size:.9rem;color:${k==='attentive'?'var(--teal)':k==='aggressive'?'var(--red)':'var(--amber)'}">${bc[k]||0}</span>
        </div>`).join('')}
    </div>
    <div style="font-size:.72rem;color:var(--slate2);">บันทึก: ${s.timestamp}</div>`;
}

function showSumModal(s){
  $('modal-title').textContent=`สรุปผล ${s.timestamp} · ${s.subject}`;
  $('modal-body').innerHTML=`<div style="font-size:.82rem;color:var(--txt2);line-height:1.9;">
    <strong>ห้อง:</strong> ${s.room} · <strong>ระยะเวลา:</strong> ${s.duration_min} นาที<br>
    <strong>คะแนนเฉลี่ย:</strong> ${s.avg_score} · <strong>สมาธิเฉลี่ย:</strong> ${s.avg_attention_min} นาที<br>
    <strong>นักเรียน:</strong> ${s.total_students} คน<br><br>
    <strong>คำแนะนำ:</strong><br>${(s.class_advice||[]).map(a=>`• ${a}`).join('<br>')}
  </div>`;
  $('sum-modal').classList.add('open');
}

async function doSummaryNow(){
  try{
    const s=await(await fetch('/api/summary/now',{method:'POST'})).json();
    renderSummaryHero(s);
    if(document.querySelector('.pg.on')?.id==='pg-summary')loadSummary();
    else{
      const btn=document.querySelector('.ni[onclick*="summary"]');
      if(btn)go('summary',btn);
    }
  }catch(e){}
}

function exportCSV(){window.location.href='/api/export/csv';}

// ── POLL ──────────────────────────────────────────────────────
loadStatus();loadEvents();
setInterval(loadStatus,2500);setInterval(loadEvents,3000);
setInterval(()=>{
  const pg=document.querySelector('.pg.on')?.id;
  if(pg==='pg-affective')loadAff();
  if(pg==='pg-health')loadHealth();
  if(pg==='pg-safety')loadSafety();
  if(pg==='pg-advice')loadAdvice();
  if(pg==='pg-summary')loadSummary();
},5000);
</script></body></html>"""

if __name__ == '__main__':
    print("=" * 60)
    print("  ระบบวิเคราะห์พฤติกรรมในห้องเรียนอัจฉริยะ v4.0")
    print("  Smart Classroom Behavioral Analysis System")
    print("  ฟีเจอร์ใหม่: คำแนะนำการเรียน + สรุปผลรายคาบ")
    print("=" * 60)
    print(f"  กล้อง {NUM_CAMERAS} ตัว | นักเรียน {NUM_STUDENTS} คน")
    print(f"  สรุปผลอัตโนมัติทุก {SUMMARY_INTERVAL//60} นาที")
    print(f"  กำลังเชื่อมต่อกล้อง...")
    init_cameras()
    active=[cid for cid,cam in cameras.items() if cam.connected]
    print(f"  กล้องที่ใช้งานได้: {active if active else 'ไม่พบ (จะแสดง OFFLINE)'}")
    print("  เริ่ม Background Analysis...")
    threading.Thread(target=background_loop,daemon=True).start()
    print("  เปิดเบราว์เซอร์: http://localhost:5000")
    print("=" * 60)
    app.run(debug=False,host='0.0.0.0',port=5000,threaded=True)