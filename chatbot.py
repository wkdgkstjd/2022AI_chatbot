import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json  #프로그램 설치

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('노잼도시 울산 되살리기 프로젝트')
st.markdown("울산경의고등학교 코드포스에서 만든 울산을 홍보하기 워한 챗봇입니다!")
st.markdown("울산의 여러 관광지들을 알려드립니다!")
st.markdown('[울산경의고등학교](http://www.gyeongui.hs.kr)')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
    
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('키워드를 입력하세요: ', '') #질문 칸
    submitted = st.form_submit_button('클릭') #전송 버튼 제작

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇/대답'])

for i in range(len(st.session_state['past'])-1,-1,-1):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
