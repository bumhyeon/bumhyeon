# python3 -m pip install pymupdf
# python3 -m pip install spacy
# python3 -m spacy download ko_core_news_sm
# python3 -m pip install tiktoken
# python3 -m pip install chromadb
# python3 -m pip install openai
# python3 -m pip install -qU langchain-openai
# pip install -U langchain-openai
# python3 -m pip install chainlit

# C:\Users\bumi\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\

# chainlit run chat_1.py --port 8500

# pip install --upgrade openai



import os #파일 경로 확인
import chainlit as cl #체인릿 사용
from langchain_openai import OpenAIEmbeddings, ChatOpenAI #chatgpt 불러오기
from langchain.document_loaders import PyMuPDFLoader #PDF 파일 올리기
from langchain.prompts import PromptTemplate #프롬프트 받기
from langchain.schema import HumanMessage #사용자메시지 받기
from langchain.text_splitter import SpacyTextSplitter #PDF파일내에 텍스트 분리
from langchain_community.vectorstores import Chroma #문서 저장 및 검색

# OpenAI 임베딩 설정
api_key = "sk-proj-PKrHrTkG6coWLfyNKXglT3BlbkFJrGLttSNNW8kiqL8ZUqsS" #OpenAI key 사용자마다 받아야함

#모델 불러오기
embeddings = OpenAIEmbeddings(

    model="text-embedding-ada-002",
    openai_api_key=api_key
)

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=api_key
)

#질의를 받아 단어 결정결정
prompt = PromptTemplate(template="""문장을 바탕으로 질문에 답하세요.
                        
문장: 
{document} 

질문: {query}
""", input_variables=['document', 'query'])

text_splitter = SpacyTextSplitter(chunk_size=300, pipeline="ko_core_news_sm") #한번의 리스트에 몇개의 단어를 받고, 한국어 NLP 모델

@cl.on_chat_start #여기서 부터 체인릿 시작
async def on_chat_start():
    files = None #파일 선택 시 시작

    while files is None:
        files = await cl.AskFileMessage(
            max_size_mb=20,
            content='PDF를 선택해 주세요',
            accept=['application/pdf'], #파일 형식
            raise_on_timeout=False, #파일 시간에 따른 실패처리안함
        ).send()

    file = files[0]

    if not os.path.exists('tmp'):
        os.mkdir('tmp') #임시 저장파일
    
    file_path = f"tmp/{file.name}"
    
    with open(file_path, "wb") as f: #읽고 쓰는 형태로 열기
        f.write(file.content)
    
    documents = PyMuPDFLoader(f"tmp/{file.name}").load() #PDF 업로드
    splitted_documents = text_splitter.split_documents(documents) #PDF내 텍스트 분할

    database = Chroma( #크로마라는 PDF저장소 생성
    embedding_function=embeddings, 
    )


    database.add_documents(splitted_documents) #분할된 문서도 데이터베이스 저장

    cl.user_session.set( #채팅사용한 곳에 저장
        "database", 
        database    
    )

    await cl.Message(content=f"'{file.name}' 로딩이 완료되었습니다. 질문을 입력하세요").send() #분할 저장완료 후 질의제공

@cl.on_message
async def on_message(input_message): #입력 메시지에 따른 호출
    print('입력된 메시지: ' + input_message)

    database = cl.user_session.get('database') #저장된 문서를 가져옴

    documents = database.similarity_search(input_message) #질의와 유사한 문서 탐색

    documents_string = "" #검색한 문서 문자열 결합

    for document in documents:
        documents_string += f"""
    ------------------------
    {document.page_content}
    """

    result = chat([ #챗봇 호출
        HumanMessage(content=prompt.format(document=documents_string, 
                                           query=input_message))
    ])

    await cl.Message(content=result.content).send() #생성된 답변제공

    # choices = result.choices
    # if choices:
    #     await cl.Message(content=choices[0].message.content).send()
    # else:
    #     await cl.Message(content="응답이 없습니다.").send()
