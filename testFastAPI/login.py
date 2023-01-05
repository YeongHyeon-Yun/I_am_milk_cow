from fastapi import APIRouter, Depends
import database
from sqlalchemy.orm import Session
from sqlalchemy import text
import json

router = APIRouter()
get_db = database.get_db


@router.get("")
def id_all(db:Session=Depends(get_db)):
    print("Test1")
    id = db.execute(text("select * from t_user")).fetchall()
    return id


@router.get('/login')
def login(id, pw, db:Session = Depends(get_db)):

    user = db.execute(text(f"SELECT user_id, user_password FROM t_user WHERE user_id = '{id}'")).fetchall()

    if user != []:
        userId = user[0][0]
        userPw = user[0][1]
        if (id == userId and pw == userPw):
            resStr = {"id":userId, "pw":userPw}
        else:
            resStr = {"id":"1", "pw":"1"}
    else:
        resStr = {"id":"1", "pw":"1"}

    print("Login request")
    return resStr

@router.post("/signUp")
def signup(id, pw, db:Session=Depends(get_db)):
    joinId = db.execute(text(f"SELECT user_id FROM t_user WHERE user_id = '{id}'")).fetchall()

    resStr = {"id":"null", "pw":"null"}
    cnt = db.execute(text(f"SELECT count(user_num) FROM t_user")).fetchall()
    userNum = cnt[0][0] + 1
    # 아이디가 기존에 없는 경우
    if joinId == []:
        print("New id")
        resStr = {"id":id, "pw":pw}
        # db에 id pw 추가
        db.execute(text(f"INSERT INTO t_user VALUES ({userNum}, '{id}', '{pw}', 'null', 'null')"))
        db.commit()
    else:
        print("Using id")
        resStr = {"id":"1", "pw":"1"}
        

    
    print(f'Check : {joinId}, {resStr}')
    print("SignUp request")
    return resStr