import pyaudio
import wave
import argparse
import speech_recognition as sr
import requests
import json
import logging
import os
from datetime import datetime
from googleapiclient.discovery import build
from llama_index import Document
from llama_index.indices import SimpleIndex
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import StorageContext
from llama_index import ServiceContext



# 设置录音参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Search API 配置
API_KEY = ''
CSE_ID = ''

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=RECORD_SECONDS, help="录音时长（秒）")
parser.add_argument("--output", type=str, default=WAVE_OUTPUT_FILENAME, help="输出文件名")
args = parser.parse_args()

# 全局搜索结果索引
index = SimpleIndex()

def google_search(query, api_key, cse_id, num_results=5):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    return res.get('items', [])

def store_search_results(results):
    documents = []
    for result in results:
        doc = Document(  # 使用 Document 类代替 SimpleDocument
            text=result.get('snippet', ''),
            metadata={
                'title': result.get('title', ''),
                'link': result.get('link', '')
            }
        )
        documents.append(doc)

    for doc in documents:
        index.add_document(doc)

def query_index(index, query):
    results = index.query(query)
    return results

def record_audio(duration, output_filename):
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        logging.info(f"开始录音，时长为 {duration} 秒...")
        frames = []

        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)

        logging.info("录音结束。")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        input_directory = "input"
        os.makedirs(input_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(input_directory, f"input_{timestamp}.wav")

        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        logging.info(f"音频文件已保存到 {output_filename}")
        return output_filename
    except Exception as e:
        logging.error(f"录音时发生错误：{e}")
        return None

def recognize_speech_from_wav(wav_filename, max_attempts=3):
    recognizer = sr.Recognizer()
    attempt = 1
    while attempt <= max_attempts:
        try:
            with sr.AudioFile(wav_filename) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="zh-CN")
                return text
        except sr.UnknownValueError:
            logging.error("无法识别音频")
        except sr.RequestError as e:
            logging.error(f"无法请求语音识别服务: {e}")
        except Exception as e:
            logging.error(f"识别时发生错误: {e}")

        attempt += 1

    logging.error(f"已达到最大尝试次数 ({max_attempts})，无法识别音频")
    return "无法识别音频"

# 定义一个搜索工具
class SearchTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="搜索", description="用于在网络上搜索信息。")

    async def _run(self, query: Query) -> str:
        # 检查 API 密钥和引擎 ID 是否设置
        if not API_KEY or not CSE_ID:
            return "请设置 Google Search API 密钥和引擎 ID。"

        # 使用 Google Search API 进行搜索
        service = build("customsearch", "v1", developerKey=API_KEY)
        try:
            results = service.cse().list(
                q=query.query_string,
                cx=CSE_ID,
            ).execute()
        except Exception as e:
            return f"搜索失败：{e}"

        # 从结果中提取相关信息
        if 'items' in results:
            search_results = "\n".join(
                [f"标题：{result['title']}\n链接：{result['link']}\n摘要：{result['snippet']}\n" for result in results['items']]
            )
            return search_results
        else:
            return "没有找到相关搜索结果。"

# 创建一个 Ollama 模型预测器
def get_ollama_response(prompt):
    try:
        url = "http://localhost:11434/api/generate"  # 替换为你的 Ollama API 地址
        headers = {"Content-Type": "application/json"}
        data = {"model": "qwen2:7b", "prompt": prompt}
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            return response.json()["response"]
        else:
            logging.error(f"Ollama API 请求失败：{response.status_code}")
            return "Ollama API 请求失败，请检查 Ollama 是否运行。"

    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama API 请求错误：{e}")
        return "Ollama API 请求错误，请检查 Ollama 是否运行。"

class OllamaPredictor(LLMPredictor):
    def __init__(self):
        super().__init__()

    async def predict(self, query: str, **kwargs) -> str:
        response = get_ollama_response(query)
        return response

llm_predictor = OllamaPredictor()
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# 创建一个 Agent
agent = OpenAIAgent(
    service_context=service_context,
    tools=[SearchTool()],
)

def synthesize_speech(text):
    url_generate = "http://127.0.0.1:5000/generate_audio"
    out_directory = "out"
    os.makedirs(out_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(out_directory, f"generated_audio_{timestamp}.wav")

    payload = {
        "text": text,
        "temperature": 0.7,
        "audio_seed_input": -1
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response_generate = requests.post(url_generate, json=payload, headers=headers)
        
        if response_generate.status_code == 200:
            data = response_generate.json()
            output_path = data.get('output_path')

            if output_path and os.path.exists(output_path):
                with open(output_filename, 'wb') as f_out, open(output_path, 'rb') as f_in:
                    f_out.write(f_in.read())

                logging.info(f"Generated audio file is saved to {output_filename}")
                return output_filename
            else:
                logging.error(f"Generated audio file path does not exist: {output_path}")
                return None

        else:
            logging.error(f"API request failed with status code {response_generate.status_code}: {response_generate.text}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred: {e}")
        return None

def play_audio_and_delete(file_path):
    try:
        wf = wave.open(file_path, 'rb')
        audio = pyaudio.PyAudio()

        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()
        audio.terminate()
        wf.close()

        logging.info(f"播放完毕，删除文件：{file_path}")
        os.remove(file_path)
    except Exception as e:
        logging.error(f"播放或删除音频文件时发生错误：{e}")

def check_for_commands(text):
    prompt = f"这是用户的输入：“{text}”。请判断这是否是一个命令，并返回命令类型（例如：“连接大模型”或“停止”），如果不是，请回答“无”。"
    response = agent.chat(prompt)
    if "大模型启动" in response:
        return "连接大模型"
    elif "停止" in response:
        return "停止"
    else:
        return "无"

def search_and_store(query):
    results = google_search(query, API_KEY, CSE_ID)
    if results:
        store_search_results(results)
        return True
    else:
        logging.error("未能获取搜索结果。")
        return False

if __name__ == "__main__":
    connected_to_model = False

    while True:
        audio_filename = record_audio(args.duration, args.output)
        if not audio_filename:
            continue
        
        recognized_text = recognize_speech_from_wav(audio_filename)
        logging.info("识别结果: %s", recognized_text)

        if recognized_text == "无法识别音频":
            continue

        if not connected_to_model:
            command = check_for_commands(recognized_text)
            if command == "连接大模型":
                connected_to_model = True
                logging.info("大模型已成功连接。")
                continue

        if connected_to_model:
            if "停止" in recognized_text:
                logging.info("检测到'停止'命令，终止程序。")
                break
            
            query = recognized_text
            if search_and_store(query):
                search_results = query_index(index, query)
                if search_results:
                    logging.info("搜索结果: %s", search_results)

                    # 将搜索结果传递给 Agent 进行处理
                    response = agent.chat(search_results)
                    logging.info("大模型回复: %s", response)

                    output_directory = os.path.dirname(audio_filename)
                    recognized_text_path = os.path.join(output_directory, "recognized_text.txt")
                    try:
                        with open(recognized_text_path, "w", encoding="utf-8") as file:
                            file.write("识别结果:\n")
                            file.write(recognized_text)
                            file.write("\n\n搜索结果:\n")
                            file.write(search_results)
                            file.write("\n\n大模型回复:\n")
                            file.write(response)
                        logging.info(f"识别结果和大模型回复已保存到 {recognized_text_path} 文件中")
                    except Exception as e:
                        logging.error(f"保存文件时发生错误：{e}")

                    response_audio_path = synthesize_speech(response)
                    if response_audio_path:
                        logging.info(f"生成的回复语音文件路径：{response_audio_path}")

                        play_audio_and_delete(response_audio_path)