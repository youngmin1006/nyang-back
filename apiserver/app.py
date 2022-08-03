import math
import os
from flask import Flask,jsonify,request
from flask_cors import CORS
from flask_mysqldb import MySQL
from dotenv import load_dotenv
import json
import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import requests
import time
from flask_mysqldb import MySQL


#모델 정의
def generate_model(
        previous_weight_path: str = "", predict: bool = False, device: str = "cpu"
) -> nn.Module:
    """KeyPointRCNN 모델을 생성, 반환합니다.

    Args:
        previous_weight_path (str, optional): 이전 가중치를 불러올 때, 가중치 경로를 입력합니다. 기본값은 ""으로, 기존 가중치를 불러오지 않습니다.
        predict (bool, optional): 모델이 평가로 사용되길 희망할 때 사용됩니다. 기본값은 False 입니다.
        device (str, optional): 모델이 사용할 디바이스를 지정합니다. 기본값은 "cpu" 입니다..
    Returns:
        nn.Module:
    """
    backbone = resnet_fpn_backbone("resnet101", pretrained=True)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )

    keypoint_roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
    )

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=aspect_ratios
    )

    model = KeypointRCNN(
        backbone,
        num_classes=2,
        num_keypoints=15,
        box_roi_pool=roi_pooler,
        keypoint_roi_pool=keypoint_roi_pooler,
        rpn_anchor_generator=anchor_generator,
    )

    if not previous_weight_path == "":
        state_dict = torch.load(previous_weight_path)
        model.load_state_dict(state_dict)

    if predict:
        model = model.eval()
    else:
        model = model.train()

    return model.to(device)


model = generate_model(
    previous_weight_path="",
    predict=True,
    device="cpu",
)
model.load_state_dict(torch.load("cpu_1.ckpt"))

def get_predict(img_path):
    #모델 돌리기 => GET 이 request 되면, 해당 사진으로 돌리기 => if 안으로 넣어주기
    img = Image.open(img_path)
    img = np.array(img)
    tensor = A.Compose([ToTensorV2()])(image=img)["image"] / 255.0
    tensor = tensor.unsqueeze(0).cpu()

    keypoints = model(tensor)[0]["keypoints"].detach().cpu().numpy().copy()[0]
    keypoints = list(map(int, np.reshape(keypoints, -1)))
    return keypoints

#각도계산
def cal_deg(point1, point2, point3):
    cx, cy = point2
    x1, y1 = point1
    x2, y2 = point3
    rad = math.atan2(y2 - cy, x2 - cx) - math.atan2(y1 - cy, x1 - cx)
    return min(
        abs(180 - abs((rad * 180) / math.pi)), abs((rad * 180) / math.pi)
    )


def cal_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    )


#유사도 검사 함수
def cal_similarty(img, points , degrees , pose):
    points.insert(0, [])
    pose = pose

    def return_condition_dict(POSE,degrees):
        condition = {}
        condition["SIT"] = {"keypoint": [[8, 4, 9], [2, 5, 14]], "range": [5, 20], "result": [16.25, 86.71],
                            "condition_num": 2}
        condition["LAY"] = {"keypoint": [[8, 5, 14], [9, 5, 14], [12, 14, 5], [13, 14, 5]], "range": [20, 20, 20, 20],
                            "result": [85.79, 69.15, 78.82, 53.69], "condition_num": 4}
        condition["BREAD"] = {"keypoint": [[6, 7, 11], [14, 13, 7]], "range": [5, 30], "result": [14.43, 89.38],
                              "condition_num": 2}
        condition[POSE]["result"] = degrees
        print(POSE," dict:", condition[POSE])
        return condition[POSE]

    # def draw_line(img, num1, num2):
    #     cv2.line(
    #         img,
    #         (points[num1][0], points[num1][1]),
    #         (points[num2][0], points[num2][1]),
    #         (0, 255, 255),
    #         5,
    #     )
    #     return img

    con = return_condition_dict(pose,degrees)
    correct_rate = 0
    for kp, ran, res in zip(con["keypoint"], con["range"], con["result"]):
        deg = cal_deg(points[kp[0]], points[kp[1]], points[kp[2]])

        # draw_line(img, kp[0], kp[1])
        # draw_line(img, kp[1], kp[2])
        # cv2.putText(
        #     img,
        #     "{0:0.2f}".format(deg),
        #     (points[kp[1]][0], points[kp[1]][1] + 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (0, 255, 0),
        #     3,
        # )

        print(deg, res, ran)
        if deg < res - ran or deg > res + ran:
            # cv2.putText(
            #     img,
            #     "FAIL",
            #     (50, 50),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1.0,
            #     (0, 0, 255),
            #     3,
            # )
            #print("조건 미충족")
            correct_rate += 0
        else:
            correct_rate += (res - abs(res - deg)) / res
    correct_rate /= con["condition_num"]
    # cv2.putText(
    #     img,
    #     "{0:0.2f}".format(correct_rate),
    #     (50, 100),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1.0,
    #     (0, 255, 0),
    #     3,
    # )

    return img, correct_rate


load_dotenv()

app = Flask(__name__)
CORS(app)

#DB 연결

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '111111'
app.config['MYSQL_DB'] = 'nyangdb'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['JSON_SORT_KEYS'] = False
db = MySQL(app)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        query = 'select * from answer'
        cur = db.connection.cursor()
        cur.execute(query)
        res = jsonify(cur.fetchall())
        return res


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        #채우기

        return "predict"
    if request.method == 'POST':
        global start
        start = time.time()
        #이미지 파일 받아오기  add 정답 이미지 정보 받아오기
        img_file = request.files['file']#도전사진 이미지 파일
        ans_id = request.form['AnsId']#문제사진 id

        keypoints = get_predict(img_file)
        points = []
        for i in range(15):
            points.append([keypoints[3*i],keypoints[3*i+1]])
        challenge_keypoint = ' '.join(map(str, keypoints))

        #문제사진 각도 가져오기 --> 각도,클래스네임 가져오기
        query = 'select degree,class_name from answer where id ='+ans_id
        cur = db.connection.cursor()
        cur.execute(query)
        db_res = cur.fetchall()
        db_res = db_res[0]
        degree_res = db_res['degree']
        degree_list = list(map(float,degree_res.split()))
        print("DB 결과",db_res)
        print("각도 결과", degree_list)

        # 유사도 계산
        print(points)
        img, simres = cal_similarty(None, points,degree_list ,db_res['class_name']) #함수에 포인트 , 각도, 클래스네임 인자 추가
        simres = int(simres*100)
        print(simres)
        return {'similar': simres}

if __name__ == '__main__':
    app.run(debug=True)