# OmniDexGrasp 실행 명령 정리 (촬영 → Stage 3)

모든 경로는 저장소 루트 `/home/unist/Paper_implementaion/OmniDexGrasp` 기준. `<task>`는 `datasets/` 아래 폴더 이름 (예: `toilet_paper_1`).

## 0. 입력 준비 — `datasets/<task>/`에 7개 파일

| 파일 | 출처 |
|---|---|
| `scene_image.png`, `depth.png`, `camera.yaml` | 1) RealSense 촬영 |
| `generated_human_grasp.png` | 2) 이미지 생성 모델 (웹 UI) |
| `base.obj`, `material.mtl`, `shaded.png` | 3) SAM 3D Objects |

### 1) 촬영
```bash
conda activate omnidexgrasp
python scripts/capture_realsense.py --task <task> --obj "<물체 설명>"   # space: 촬영, q: 종료
```
- 물체를 화면 중앙, 40~50 cm 거리에, 가리는 것 없이.
- `--obj`는 GSAM 프롬프트가 됨. 짧은 명사구 (예: `tissue roll`). Stage 1 후 확인.

### 2) 사람 손 이미지
`scene_image.png`를 Gemini 또는 ChatGPT에 올리고 `omnidexgrasp/scripts/gen_human_grasp.py`의 `BASE_PROMPT`를 붙여넣어 생성. 원본과 같은 4:3, RGB PNG로 저장.
```bash
python -c "from PIL import Image; Image.open('/home/unist/Downloads/result.png').convert('RGB').save('datasets/<task>/generated_human_grasp.png')"
```

### 3) 물체 메쉬 (SAM 3 → SAM 3D Objects)
```bash
# 마스크
cd ../sam3 && PYTHONPATH=$PWD /home/unist/anaconda3/envs/sam3/bin/python \
  ../OmniDexGrasp/scripts/sam3_mask.py ../OmniDexGrasp/datasets/<task>/scene_image.png "<물체 설명>" /tmp/mask.png
# 메쉬 (16GB GPU, 약 1분)
cd ../sam-3d-objects && /home/unist/anaconda3/envs/sam3d-objects/bin/python demo_low_vram.py \
  --image ../OmniDexGrasp/datasets/<task>/scene_image.png --mask /tmp/mask.png --output /tmp/raw.obj
# 6만 면으로 축소 + base.obj/material.mtl/shaded.png 생성 + 확인용 렌더
cd ../OmniDexGrasp && conda activate omnidexgrasp && \
  python scripts/sam3d_to_base_obj.py /tmp/raw.obj datasets/<task> /tmp/render
```

## Stage 1 — 복원 (저장소 루트에서)
```bash
bash scripts/run_reconstruction.sh sequential 'tasks=[<task>]'   # GSAM → HaMeR 순차 (16GB용)
bash scripts/run_reconstruction.sh pose 'tasks=[<task>]'         # MegaPose 물체 자세
```
확인 (오류가 있어도 로그는 `Failed: 0`으로 끝날 수 있으므로 직접 본다):
```bash
python -c "import trimesh; print(trimesh.load('/home/unist/Paper_implementaion/OmniDexGrasp/out/<task>/scaled_mesh.obj', force='mesh').extents.round(3))"   # m 단위, 실측과 비교
```
- `out/<task>/data/recons/vis/gsam_scene_annotated.jpg` 박스가 물체만 감싸는지
- `out/<task>/data/recons/vis/pose_scene.jpg`, `pose_grasp.jpg`에서 렌더가 물체에 겹치는지

| 증상 | 다시 할 것 |
|---|---|
| 박스가 물체 외 영역까지 감쌈 | `camera.yaml`의 `obj_description` 수정 → `sequential` → `pose` |
| 박스는 맞는데 크기가 틀림 | 메쉬 재생성(0단계 3) → `sequential` → `pose` |
| 크기는 맞는데 렌더가 어긋남 | `pose`만 |

이미 Stage 2·3을 돌렸다면 그것도 다시 실행한다. (`toilet paper`는 `toilet`으로 잘려 책상 전체가 잡혔고, `tissue roll`로 바꿔 해결.)

## Stage 2 — 손 자세 최적화
```bash
cd omnidexgrasp && conda activate omnidexgrasp && export PYTHONNOUSERSITE=1
python -m optim.main 'tasks=[<task>]'          # → out/<task>/optim_res.json, optim_res.ply
```

## Stage 3 — 로봇손 리타게팅
```bash
python -m human2robo.main 'tasks=[<task>]'     # → out/<task>/robo.json (hand_types 기본 [inspire])
python -m scripts.vis_dexgrasp --output ../out --port 8080   # 브라우저 확인
```

## 로봇 실행 (xarm-teleop 환경: xArm SDK 포함)

### 1) 카메라-베이스 캘리브레이션 (카메라를 고정한 뒤 1회)
팔 끝(손등)에 ChArUco 보드를 붙이고, 보드가 잘 보이는 기준 자세의 관절값 7개를 넣는다. 손목 관절만 바꾼 20자세를 자동으로 돌며 기록·풀이한다.
```bash
conda activate xarm-teleop
python scripts/calibrate_eye_to_hand.py --base <J1> <J2> <J3> <J4> <J5> <J6> <J7>   # → calibration/eye_to_hand.json
```
확인: `consistency std` 5 mm 이하. 이후 카메라를 절대 움직이지 않는다 (움직이면 재캘리브레이션 + 재촬영).

### 2) 캘리브레이션 검증 (Stage 1 이후)
```bash
python scripts/verify_calibration.py --task <task>          # 물체 중심의 베이스 좌표 출력
python scripts/verify_calibration.py --task <task> --move   # TCP를 물체 중심 15 cm 위로 이동
```
손이 물체 바로 위에 오면 통과.

### 3) 손 장착 변환
`calibration/hand_mount.json`에 플랜지→손 URDF 루트 변환(yaw, 어댑터 두께). 현재 값은 카메라 실루엣 피팅으로 추정(yaw 215°, d 0 mm). 장착을 바꾸면 수정.

### 4) 목표 자세 계산 → 실행
```bash
cd omnidexgrasp && conda activate omnidexgrasp
python -m scripts.robo_to_camera --task <task>   # → robo_cam.json: T_cam_hand, T_base_hand, T_base_tcp, xarm_tcp_aa_mm_rad
cd .. && conda activate xarm-teleop
python scripts/execute_grasp.py --task <task> --lift 0.03   # 사전자세 → 파지자세 → 손가락 닫기 → 들어올리기, 단계마다 Enter
python scripts/grasp_from_robo.py out/<task>/robo.json      # 손만 따로 쥐어 볼 때 (--close-scale 1.3 등으로 더 닫기)
```
- `--lift`: 사람 손 자세가 책상에 너무 낮을 때 전체를 올리는 보정(m). `--up`: 사전 자세 높이(기본 8 cm, 파지 자세 바로 위). `--back`: 손가락 반대 방향 후퇴(기본 0; 이 파지는 베이스 쪽이라 쓰지 않음).
- 실행 전 비상정지에 손을 올리고, 첫 실행은 `--speed 20` 정도로.
