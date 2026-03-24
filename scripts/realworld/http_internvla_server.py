import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''


@app.route("/eval_dual", methods=['POST']) #클라이언트가 요청을 보내면 이 함수가 실행된다. 
def eval_dual():
    global idx, output_dir, start_time
    start_time = time.time()

    #input은 image(RGB, Depth)
    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)

    #RGB 이미지 처리 
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    #Depth 이미지 처리 
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}") #요청 받은 뒤 json 파싱까지 걸린 시간 출력함 

    #camera pose, 기본 좌표계 
    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    #현재 수행할 자연어 명령어 하드코딩 되어있음, 서버는 아무 instruction이나 받는 구조가 아니고, 항상 이 문장 하나만 수행함 
    instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
    
    #reset 처리, 에이전트 내부 상태 초기화용 
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    # 매 요청마다 step count를 1 증가 
    idx += 1

    #look down 플래그 
    look_down = False
    t0 = time.time()
    dual_sys_output = {}

    # 입력:이미지,카메라 pose, 자연어 명령, 카메라 내부 파라미터, 제어 플레그 -> 출력: dual_sys_output
    #agent -> step check
    dual_sys_output = agent.step(
        image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
    )

    # action ->5 일 시, 특수 신호로 보고 다시 한 번 추론함 
    if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
        look_down = True
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )

    #최종 응답 형식 만들기 
    json_output = {}

    #discreteaction 이 있으면 action 반환 
    if dual_sys_output.output_action is not None:
        json_output['discrete_action'] = dual_sys_output.output_action
    
    #action이 없으면 trajectory 반환함 
    else:
        json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
        if dual_sys_output.output_pixel is not None:
            json_output['pixel_goal'] = dual_sys_output.output_pixel

    #추론 시간 출력 
    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")

    #클라이언트에게 응답 반환
    return jsonify(json_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    args = parser.parse_args()

    # 카메라 내부 파라미터 행렬 
    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    #agent 모델 로드 
    agent = InternVLAN1AsyncAgent(args)

    # warming-up 용도 
    agent.step(
        np.zeros((480, 640, 3)),
        np.zeros((480, 640)),
        np.eye(4),
        "hello",
    )
    agent.reset()

    #Flask 서버 실행 
    app.run(host='0.0.0.0', port=5801)
