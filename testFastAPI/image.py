from tkinter import Image
from fastapi import Depends, FastAPI, APIRouter, File, Form, Response, UploadFile
import database
from sqlalchemy.orm import Session
from sqlalchemy import text
import json
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form
import multipart
# from uuid import UUID
from tempfile import NamedTemporaryFile
from typing import IO
import uuid, os
import argparse
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
from random import randint
import base64
import sys
sys.path.append('/workspace')
sys.path.append('/workspace/dusik')
sys.path.append('/workspace/bum')
# print(sys.path)
from dusik.cow_detect.is_cow import find_cow
from bum.MetricLearningIdentification.only_test import evaluateModel
from bum.MetricLearningIdentification import *

router = APIRouter()
get_db = database.get_db
cow_index = 30
router.mount("/workspace/testFastAPI/images", StaticFiles(directory="/workspace/testFastAPI/images"), name="images")

 # 이미지를 저장할 서버 경로
IMAGEDIR = "/workspace/testFastAPI"

@router.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

# 이미지 하나만 저장 / 리턴값 상의
@router.post("/cowImgUp")
async def upload_cow_img(file: UploadFile = File(...)):
    res = "none"
    # 파일 이름 id값으로 저장
    # 0번이 대표 이미지
    file.filename = "0.jpg"
    contents = await file.read()  # <-- Important!
    with open(f"{IMAGEDIR}/test1/0.jpg", "wb") as f:
        f.write(contents)
    flag = 0   
    find_cow_result = find_cow(flag)
    if (find_cow_result == False):
        print(f"소 아님: {find_cow_result}")
        res = "false"
        return res
    else:
        # 소라면
        print('소 확인!')
        setting_args = setup_args()
        recog_boolean = True
        regist_boolean = False
        result_label = evaluateModel(setting_args, recog_boolean, regist_boolean, flag)
        print(result_label)
        
        if (result_label == None):
            # 미등록개체 / 등록여부
            print("미등록 개체, 등록할래?")
            res = "true"
        else:
            res = f"{result_label}"
            print(f"등록개체 : {result_label}")
        
        # return {"filename": file.filename}
    print(res)
    # return 'success'
    return res

# 이미지 여러장 저장 / 리턴값 상의
@router.put("/cowImgList")
async def upload_cow_imglist(files: List[UploadFile] = File(...),
):
    print("이미지 리스트 요청")
    path=f"{IMAGEDIR}/test2"

    # # id 파일이 있으면
    # if (os.path.isdir(f"{IMAGEDIR}")):
    #     # id 파일이 있으면
    #     if (os.path.isdir(path)):
    #         pass
    #     else:
    #         os.mkdir(path)
    # else:
    #     # id파일 없으면 폴더 생성
    #     os.mkdir(f"{IMAGEDIR}")
    #     os.mkdir(path)

    # 파일을 순서대로 저장
    for i in range(len(files)):
        files[i].filename = f"{i}.jpg"
        print(files[i])

        contents = await files[i].read() 
        with open(f"{path}/{files[i].filename}", 'w+b') as buffer:
            buffer.write(contents)

    # 리스트
    flag = 1
    find_cow_result = find_cow(flag)
    if (find_cow_result == False):
        print(f"소 아님: {find_cow_result}")
        res = "false"
        return res
    else:
        # 소라면
        print('소 확인!')
        setting_args = setup_args()
        recog_boolean = True
        regist_boolean = False
        result_label = evaluateModel(setting_args, recog_boolean, regist_boolean, flag)
        print(result_label)
        if (result_label == None):
            # 미등록개체 / 등록여부
            # print("미등록 개체, 등록할래?")
            res = "true"
        else:
            res = f"{result_label}"
            print(f"등록개체 : {result_label}")
            # res = "false"
            print("등록하려는 이미지 중 기존에 등록된 개체가 확인됩니다.")

    # return {"filenames": [file.filename for file in files]}
    return res




# 이미지 내보내기
@router.get("/cowImgOut")
async def cow_img_out(cow_id:str):
    print("이미지 요청")
    path = f"{IMAGEDIR}/images/{cow_id}/0.jpg"
    return FileResponse(path)




# 소 전체 이미지
@router.get("/cowsImages")
async def cow_img_list_out(user_id: str, db:Session=Depends(get_db)):
    print("이미지 전체 요청")
    cow_id = db.execute(text(f"SELECT cow_id FROM t_cow where user_num = (select user_num from t_user where user_id = '{user_id}')")).fetchall()
    print(len(cow_id))
    files = os.listdir(f"{IMAGEDIR}/images")

    path_list = []
    for i in range(1, len(cow_id)+1):
        id = 99
        id += i
        path_list.append(f"{IMAGEDIR}/images/{i}.jpg")
    cow_id.append(path_list)

    files = File(path_list)




    return  files



def setup_args():
    parser = argparse.ArgumentParser(description='Params')

	# Required arguments
    parser.add_argument('--model_path', nargs='?', type=str, default="/workspace/bum/MetricLearningIdentification/output/fold_0/best_model_state.pkl", 
						help='Path to the saved model to load weights from')
    parser.add_argument('--folds_file', type=str, default="/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/splits/10-90-custom.json",
						help="The file containing known/unknown splits")
    parser.add_argument('--save_path', type=str, default='/workspace/bum/MetricLearningIdentification/output/fold_0',
						help="Where to store the embeddings")

    parser.add_argument('--dataset', nargs='?', type=str, default='only_test_OpenSetCows2020', 
						help='Which dataset to use')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
    parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='Size of the dense layer for inference')
    parser.add_argument('--current_fold', type=int, default=0,
						help="The current fold we'd like to test on")
    parser.add_argument('--save_embeddings', type=bool, default=True,
						help="Should we save the embeddings to file")
    args = parser.parse_args()
    
    return args