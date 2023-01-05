import os
from fastapi import APIRouter, Depends
import database
from sqlalchemy.orm import Session
from sqlalchemy import text
import json
from typing import Optional, List
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
import sys
import argparse
sys.path.append('/workspace')
sys.path.append('/workspace/bum')
from bum.MetricLearningIdentification.only_test import evaluateModel
from bum.MetricLearningIdentification import *


router = APIRouter()
get_db = database.get_db

 # 이미지를 저장할 서버 경로
IMAGEDIR = "Files/images"
class t_cow(BaseModel):
  cow_id: str
  cow_name: str
  cow_birth: str 
  cow_variety: str
  cow_gender: str
  cow_vaccination: str
  cow_pregnancy: str
  cow_milk: str
  cow_castration: str
  wish_list: int
  user_num: int
  class Config:
      orm_mode = True


@router.get("/")
def test():
    print("/info")
    return {"message" : "hehehe...."}

# 전체 정보 보내기
@router.get("/cowInfoAll")
async def read_item_all(user_id:str, db:Session=Depends(get_db)):
    print("전체 개체 정보 요청")
    # t_cow = db.execute(text(f"SELECT * FROM t_cow where ")).fetchall()
    t_cow = db.execute(text(f"SELECT * FROM t_cow WHERE user_num = (SELECT user_num FROM t_user WHERE user_id = '{user_id}')")).fetchall()
    print(len(t_cow))
    print(t_cow[0])
    return t_cow

# 원하는 하나의 개체 정보 보내기
@router.get("/cowInfoOne")
async def read_item(cow_id:str, db:Session=Depends(get_db)):
    print(f"{cow_id}번 개체 정보 요청")
    try:
        # 개채 있으면
        t_cow = db.execute(text(f"SELECT * FROM t_cow WHERE cow_id = {cow_id}")).fetchall()
    except:
        # 개채 없으면
        t_cow = None
    print(t_cow)
    return t_cow

# 즐겨찾기 개체 정보 내보내기
@router.get("/cowInfoWish")
async def read_item_all(user_id:str, db:Session=Depends(get_db)):
    print("전체 개체 정보 요청")
    # t_cow = db.execute(text(f"SELECT * FROM t_cow where ")).fetchall()
    t_cow = db.execute(text(f"SELECT * FROM t_cow WHERE wish_list = 1")).fetchall()
    print(len(t_cow))
    print(t_cow[0])
    return t_cow



# cow 정보 받아 등록하기
@router.post("/cowInfoRegist")
async def cow_info_regist(user_id:str, item: t_cow, db:Session=Depends(get_db)):
    print("info regist check")
    print(item)
    # item = db.execute(text(f"SELECT * FROM info WHERE id == {id}")).fetchall()
    # db.execute(item.dict(exclude_unset=True))
    cow_name = item.cow_name
    cow_birth = item.cow_birth
    cow_variety = item.cow_variety
    cow_gender = item.cow_gender
    cow_vaccination = item.cow_vaccination
    cow_pregnancy = item.cow_pregnancy
    cow_milk = item.cow_milk
    cow_castration = item.cow_castration
    wish_list = item.wish_list
    user_num = db.execute(text(f"(SELECT user_num FROM t_user WHERE user_id = '{user_id}')")).fetchall()

    try:
        db.execute(text(f"INSERT INTO t_cow (cow_name, cow_birth, cow_variety, cow_gender, cow_vaccination, cow_pregnancy, cow_milk, cow_castration, wish_list, user_num) VALUES ('{cow_name}','{cow_birth}','{cow_variety}','{cow_gender}','{cow_vaccination}','{cow_pregnancy}','{cow_milk}','{cow_castration}','{wish_list}','{user_num[0][0]}')"))
        db.commit()
        res = "success"
    except:
        res = "fail"
    
    if res == "success":
        print('================================================success')
        setting_args = setup_args()
        recog_boolean = False
        regist_boolean = True
        flag = 1
        evaluateModel(setting_args, recog_boolean, regist_boolean, flag)
    
    return res

# cow 정보 수정하기
@router.put("/cowInfoUpdate")
async def cow_info_update(cow_id:str, item: t_cow, db:Session=Depends(get_db)):
    print("update check")
    # db.execute(item.dict(exclude_unset=True))
    # cow_id, user_id는 제외
    cow_name = item.cow_name
    cow_birth = item.cow_birth
    cow_variety = item.cow_variety
    cow_gender = item.cow_gender
    cow_vaccination = item.cow_vaccination
    cow_pregnancy = item.cow_pregnancy
    cow_milk = item.cow_milk
    cow_castration = item.cow_castration
    wish_list = item.wish_list

    try:
        db.execute(text(f"UPDATE t_cow SET cow_name = '{cow_name}', cow_birth = '{cow_birth}', cow_variety = '{cow_variety}', cow_gender = '{cow_gender}', cow_vaccination = '{cow_vaccination}', cow_pregnancy = '{cow_pregnancy}', cow_milk = '{cow_milk}', cow_castration = '{cow_castration}', wish_list = {wish_list} WHERE cow_id = {cow_id}"))
        db.commit()
        res = "success"
    except:
        res = "fail"

    return res

# cow 정보 삭제
@router.delete("/cowInfoDelete")
async def cow_info_delete(cow_id:str, db:Session=Depends(get_db)):
    print("delete cow info")
    # db.execute(item.dict(exclude_unset=True))

    try:
        db.execute(text(f"DELETE FROM t_cow WHERE cow_id = {cow_id}"))
        db.commit()
        res = "success"
    except:
        res = "fail"

    return res



# 마이페이지 정보 요청
@router.get("/myPageInfo")
async def mypage_item(user_id:str, db:Session=Depends(get_db)):
    print("전체 개채 수, 신생우, 암/숫소")

    # t_cow = db.execute(text(f"SELECT * FROM t_cow where ")).fetchall()
    user_farmname = db.execute(text(f"SELECT user_farmname FROM t_user WHERE user_id = '{user_id}'")).fetchall()
    user_phone = db.execute(text(f"SELECT user_phone FROM t_user WHERE user_id = '{user_id}'")).fetchall()
    totalCow = db.execute(text(f"SELECT COUNT(cow_id) FROM t_cow WHERE user_num = (SELECT user_num FROM t_user WHERE user_id = '{user_id}')")).fetchall()
    babyCowCount = db.execute(text(f"SELECT COUNT(cow_id) FROM t_cow WHERE cow_birth > (SELECT DATE_ADD(NOW(), INTERVAL -6 MONTH))")).fetchall()
    cow = db.execute(text(f"SELECT COUNT(cow_gender) FROM t_cow WHERE cow_gender = '암컷'")).fetchall()
    bull = totalCow[0][0] - cow[0][0]
    print(totalCow[0][0], babyCowCount[0][0], cow[0][0], bull)


    myPageInfo = { 'user_farmname' : user_farmname[0][0], 'user_phone' : user_phone[0][0], 'totalCow' : totalCow[0][0], 'babyCowCount' : babyCowCount[0][0], 'cow' : cow[0][0], 'bull' : bull}
    print(myPageInfo)
    return myPageInfo


# wish_list 통신
@router.get("/cowWish")
async def cow_wish(cow_id: str, db:Session=Depends(get_db)):
    print(f"{cow_id}번 소 wish 요청")
    
    wishNum = db.execute(text(f"SELECT wish_list FROM t_cow WHERE cow_id = {cow_id}")).fetchall()
    print(wishNum[0][0])
    
    if (wishNum[0][0] == 0):
        # 즐찾 추가
        res = 1
    else:
        # 즐찾 해제
        res = 0
        
    db.execute(text(f"UPDATE t_cow set wish_list = {res} WHERE cow_id = {cow_id}"))
    db.commit()

    return res


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





