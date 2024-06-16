from datetime import datetime, timedelta
from io import BytesIO
import os
import textwrap
from openai import OpenAI
import streamlit as st
import requests
import openai
from bs4 import BeautifulSoup
from newspaper import Article, Config
import time
from PIL import Image
from newspaper.article import ArticleException
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader, select_autoescape


#환경변수 읽기
from dotenv import load_dotenv
load_dotenv()

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image and Text Display</title>
    <style>
        .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }
        .image-text-pair {
            display: flex;
            margin: 20px;
        }
        .image {
            margin-right: 20px;
            width: 150px;  /* 이미지의 너비를 150px로 설정 */
            height: 100px; /* 이미지의 높이를 100px로 설정 */
        }
        .text {
            font-size: 16px;
        }
        .text.first {
            font-weight: bold;
            font-style: italic;
            font-size: 18px;
        }
        .link {
            color: blue;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        {% for image_path, texts in image_text_pairs %}
        <div class="image-text-pair">
            <img class="image" src="{{ image_path }}" alt="Image">
            <div class="text-container">
                {% for idx, text in enumerate(texts) %}
                <div class="text {% if idx == 0 %}first{% endif %}">
                    {% if idx == 3 %}
                    <a href="{{ text }}" class="link">Read More</a>
                    {% else %}
                    {{ text }}
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""
        
def save_as_html(filename='output.html'):
    # Jinja 환경 설정    
    env = Environment(
        loader=FileSystemLoader('.'),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.globals['enumerate'] = enumerate  # enumerate 추가
    
    # 템플릿 로드
    template = env.from_string(html_template)
    
    html_content = template.render(image_text_pairs=st.session_state.image_text_pairs)
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(html_content)
        
def is_english_text(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def translate(text, source_lang, target_lang):
    """
    OpenAI API를 사용하여 텍스트를 번역합니다.
    
    :param text: 번역할 텍스트
    :param source_lang: 원본 언어 코드 (예: 'ko' for Korean)
    :param target_lang: 대상 언어 코드 (예: 'en' for English)
    :return: 번역된 텍스트
    """
    try:
        client = OpenAI()
        response = client.completions.create(
          model="gpt-3.5-turbo-instruct",  # 사용할 GPT 모델을 지정합니다. 가장 최신 모델을 확인하고 적용하세요.
          prompt=f"Translate the following text from {source_lang} to {target_lang} with five key bullet points::\n\n{text}",
          temperature=0.3,
          max_tokens=300,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0
        )
        
        # 응답에서 번역된 텍스트를 가져옵니다.
        # pprint(response)
        translated_text = response.choices[0].text.strip()
        # pprint(translated_text)
        return translated_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def makeImageTextPairs(index, new_image_path, new_text1,new_text2,new_text3,read_more):
    new_pair = (new_image_path, [new_text1, new_text2, new_text3, read_more])
    # pprint("makeImageTextPairs :" + new_text1)
    # global image_text_pairs
    st.session_state.image_text_pairs.insert(index, new_pair)

def create_hyperlink(url, link_text):
    """
    주어진 URL로 이동하는 하이퍼링크가 포함된 텍스트를 생성합니다.
    
    :param url: 이동할 URL 주소
    :param link_text: 링크에 표시될 텍스트
    :return: HTML 형식의 하이퍼링크 텍스트
    """
    # HTML 링크를 생성하여 반환합니다.
    hyperlink = f'<a href="{url}" target="_blank">{link_text}</a>'
    return hyperlink

def get_webpage_content(index, url, config, important_keywords = None):
    # Article 객체 생성
    try:
        article = Article(url, config=config)
        article.download()
        while article.download_state == 0: #ArticleDownloadState.NOT_STARTED is 0
            time.sleep(1)
        article.parse()
    except ArticleException as e:
        print(f"Article download failed for {url}: {e}")
        return "Failure"  # 실패시 Failure 반환

    if important_keywords is None:
        # 요약과 번역 실행
        summary_text = summarize_text(article.text)
        if is_english_text(summary_text):
            src_lang = 'en'
            dst_lang = 'ko'
        else:
            src_lang = 'ko'
            dst_lang = 'en'
        translated_text = translate(summary_text, src_lang, dst_lang)

        # 이미지와 텍스트 쌍 생성
        makeImageTextPairs(index, article.top_image, article.title, summary_text, translated_text, url)
        return "Success"  # 성공적으로 처리된 경우 Success 반환
    
    # 중요 키워드가 article.text에 포함되어 있는지 검사
    if any(keyword.lower() in article.text.lower() for keyword in important_keywords):
        # 요약과 번역 실행
        summary_text = summarize_text(article.text)
        if is_english_text(summary_text):
            src_lang = 'en'
            dst_lang = 'ko'
        else:
            src_lang = 'ko'
            dst_lang = 'en'
        translated_text = translate(summary_text, src_lang, dst_lang)

        # 이미지와 텍스트 쌍 생성
        makeImageTextPairs(index, article.top_image, article.title, summary_text, translated_text, url)
        return "Success"  # 성공적으로 처리된 경우 Success 반환
    else:
        return "No Important Keywords"  # 중요 키워드가 없는 경우
    
def summarize_text(text):
    """텍스트를 요약합니다."""
    client = openai.Client()  # API 클라이언트 초기화 (API 키 설정 필요)
    
    # 기본 프롬프트 설정
    prompt_text = "Summarize the following text into five key bullet points:\n\n"
    
    # 텍스트가 최대 문자 길이를 초과하는지 확인하고 필요시 잘라냅니다.
    max_length = 3000  # 이 값은 경험적으로 조정할 수 있습니다.
    if len(text) > max_length:
        text = textwrap.shorten(text, width=max_length, placeholder="...")

    # 최종 프롬프트 생성
    full_prompt = f"{prompt_text}{text}"
    
    # OpenAI API를 통해 텍스트 요약
    response = client.completions.create(
      model="gpt-3.5-turbo-instruct",  # 사용할 GPT 모델을 지정합니다. 가장 최신 모델을 확인하고 적용하세요.
      prompt=full_prompt,
      temperature=0.7,
      max_tokens=100,  # 요약의 최대 길이를 지정합니다.
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response.choices[0].text.strip()  # 결과 반환

def get_google_news_links(start_date, end_date, search_query, language = 'en'):
    # 구글 뉴스 검색 URL 생성
    base_url = "https://news.google.com/search"
    # start_date = datetime.strptime(start_date, "%Y-%m-%d")
    # end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = (end_date - start_date).days
    
    if language == 'en':
        hl_pararm = 'en-US'
        gl_param = 'US'
        ceid_param = 'US:en'
    else:
        hl_pararm = 'ko-KR'
        gl_param = 'KR'
        ceid_param = 'KR:ko'
    
    for offset in range(date_range + 1):
        current_date = start_date + timedelta(days=offset)
        formatted_date = current_date.strftime("%Y-%m-%d")
        params = {
            'q': search_query,
            'hl': hl_pararm,   # 언어 설정
            'gl': gl_param,      # 국가 설정
            'ceid': ceid_param, # 뉴스 발행 국가 및 언어
            'tbs': f'cdr:1,cd_min:{formatted_date},cd_max:{formatted_date}' # 날짜 범위
        }
        
        # 요청 및 응답
        response = requests.get(base_url, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 뉴스 항목 추출
        news_links = []
        seen_links = set()  # 이미 본 링크를 추적하기 위한 세트
        for a in soup.find_all('a', href=True):
            if len(news_links) >= 30:
                break  # 링크 개수가 30개에 도달하면 내부 루프도 중단
            href = a['href']
            if href.startswith('./articles/'):
                full_url = f'https://news.google.com{href[1:]}'
                # 링크가 이미 본 적이 없는 경우에만 추가
                if full_url not in seen_links:
                    news_links.append(full_url)
                    seen_links.add(full_url)  # 링크를 '본 것'으로 표시
    return news_links
        
def get_final_url(url):
    try:
        response = requests.get(url, allow_redirects=True)
        # 최종 URL을 반환합니다.
        return response.url
    except requests.RequestException as e:
        print(f"URL을 가져오는 중 오류 발생: {e}")
        return None

def get_news_content(news_links, number):
    count = 0
    
    # User-Agent 설정을 위한 Config 객체 생성
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

    # 캐싱 활성화 (기본값은 True이므로 명시적으로 설정할 필요는 없지만, 참고를 위해 추가)
    config.memoize_articles = True
    candidate_links = []  # 이곳에 대기 중인 후보 링크를 저장
    # image_text_pairs = [] # 이미지와 텍스트 쌍을 저장하는 리스트
    
    for link in news_links:
        if (count < number):
            # 링크에서 따옴표를 제거합니다.
            cleaned_link = link.strip("'")
            final_link = get_final_url(cleaned_link)
            # pprint(final_link)
            result = get_webpage_content(count, final_link, config, priority_list)
            # pprint(content)
            if (result == 'Success'):
                count += 1
            else:
                candidate_links.append(final_link)

    # count가 number에 도달하지 못한 경우, candidate_links에서 추가 작업 수행
    for link in candidate_links:
        if count < number:
            result = get_webpage_content(count, link, config)
            if result == 'Success':
                count += 1
        else:
            break  # 이미 필요한 수량을 충족한 경우 루프 종료
    # pprint(st.session_state.image_text_pairs)
    return st.session_state.image_text_pairs  # 이 부분은 함수가 마지막 링크의 컨텐츠만 반환한다는 점을 고려해야 합니다.

def mainDisplay():
    # 열마다 이미지와 텍스트 3개가 쌍으로 이루어지도록 표시
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    # pprint(image_text_pairs)
    
    for index, (image_path, texts) in enumerate(st.session_state.image_text_pairs):
        # Streamlit 컬럼 생성
        col1, col2 = st.columns([1, 3])  # 이미지는 더 작은 비율의 열에, 텍스트는 더 큰 비율의 열에 할당        
        with col2:
            # 첫 번째 텍스트 항목에 대해 큰 굵은 이탤릭체 적용
            for idx, text in enumerate(texts):
                if idx == 0:  # 첫 번째 텍스트에만 적용
                    st.markdown(f"<h2 style='font-weight:bold; font-style:italic;'>{text}</h2>", unsafe_allow_html=True)
                elif idx == 3: # URL link 만들기
                    st.markdown(create_hyperlink(text, 'read more'), unsafe_allow_html=True)
                else:
                    st.write(text)
        with col1:
            for _ in range(5):  # 예를 들어 5개의 공간을 추가
                st.write("")  # 스페이스 추가
                
            try:
                # 이미지 URL에서 이미지를 로드합니다.
                response = requests.get(image_path, headers=headers)
                image = Image.open(BytesIO(response.content))
                st.image(image)
            except requests.exceptions.RequestException as e:
                st.error(f"Requests error: {e}")
            except IOError as e:
                st.error(f"PIL error: {e}")

        # 마지막 항목이 아닌 경우에만 구분선 추가
        if index < len(st.session_state.image_text_pairs) - 1:
            st.markdown("---")  # 구분선 추가

def runningTask(keyWords):
    # result = keyWords.title()
    if is_english_text(keyWords.title()):
        news_links = get_google_news_links(fromDate, toDate, keyWords.title())
    else:
        news_links = get_google_news_links(fromDate, toDate, keyWords.title(), "ko")
        # pprint(news_links)
    result = get_news_content(news_links, number)
    # st.sidebar.success(result)
 
def doRunningTask():
    # 버튼 추가 상태를 True로 설정
    global printedImageText
    with st.spinner('Wait for it...'):
        printedImageText = runningTask(keyWords)
    st.session_state.button_clicked = True

st.title("Search News")

# Text Input
keyWords = st.sidebar.text_input("Enter keywords", "Type Here ...")

# 사이드바에서 사용자 입력 받기
input_string = st.sidebar.text_input("Enter priority separated by commas", "Samsung,Apple,MS,OpenAI")

# 문자열을 쉼표로 구분하여 리스트로 변환
priority_list = [priority.strip() for priority in input_string.split(',') if priority.strip()]

#날짜 입력
fromDate = st.sidebar.date_input("From")
toDate = st.sidebar.date_input("To")

# 라디오 버튼을 사용하여 선택지 생성
option = st.sidebar.radio(
     "Number of News",
     ('5', '10', '20'))
number = int(option)
# image_text_pairs = []
# printedImageText = []

if 'image_text_pairs' not in st.session_state:
    st.session_state.image_text_pairs = []
    
# if st.sidebar.button("Search", key='keywords'):
#     with st.spinner('Wait for it...'):     
#         printedImageText = runningTask(keyWords)
#         st.success('done')

# if st.sidebar.button("download"):
#      save_as_html(printedImageText)

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
        
# 첫 번째 버튼을 사이드바에 추가
if st.sidebar.button('Search', on_click=doRunningTask):
    mainDisplay()
    pass  # 버튼이 클릭되면 add_button 함수가 호출됨

# button_clicked 상태가 True이면 새로운 버튼 추가
if st.session_state.button_clicked:
    if st.sidebar.button('download'):
        save_as_html()
        mainDisplay()
