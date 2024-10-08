import requests
from typing import Optional, List, Dict, Any
import time

from settings import DASHSCOPE_API_KEY


class APIException(Exception):
    pass


class InvalidApiKeyException(APIException):
    pass


class JobFailedException(APIException):
    def __init__(self, code: str, message: str):
        super().__init__(f"作业失败，错误代码: {code}, 消息: {message}")
        self.code = code
        self.message = message


def _handle_error(err_response: requests.Response):
    try:
        resp_json = err_response.json()
        code = resp_json.get("code", "UnknownError")
        message = resp_json.get("message", "未提供错误消息。")
        if code == "InvalidApiKey":
            raise InvalidApiKeyException(message)
        else:
            raise APIException(f"错误代码: {code}, 消息: {message}")
    except ValueError:
        # 响应不是JSON格式，重新抛出HTTP错误
        err_response.raise_for_status()


class ImageGenSDK:

    def __init__(self):
        self.api_key = DASHSCOPE_API_KEY
        self.base_url = "https://dashscope.aliyuncs.com/api/v1".rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def submit_text2image_job(self, prompt: str) -> str:
        url = f"{self.base_url}/services/aigc/text2image/image-synthesis"
        headers = self.headers.copy()
        headers["X-DashScope-Async"] = "enable"
        data = {
            "model": "wanx-v1",
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "style": "<auto>",
                "n": 1,
            }
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            _handle_error(response)

        resp_json = response.json()
        if "code" in resp_json and resp_json["code"] == "InvalidApiKey":
            raise InvalidApiKeyException(resp_json["message"])

        task_id = resp_json.get("output", {}).get("task_id")
        if not task_id:
            raise APIException("提交作业后未返回 task_id。")
        return task_id

    def check_job_status(self, task_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/tasks/{task_id}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            _handle_error(response)

        resp_json = response.json()
        if "code" in resp_json and resp_json["code"] == "InvalidApiKey":
            raise InvalidApiKeyException(resp_json["message"])

        return resp_json

    def get_job_result(self, task_id: str, wait: bool = False, poll_interval: int = 3, timeout: int = 300) -> Optional[List[str]]:
        start_time = time.time()
        while True:
            status_response = self.check_job_status(task_id)
            output = status_response.get("output", {})
            task_status = output.get("task_status")

            if task_status == "SUCCEEDED":
                results = output.get("results", [])
                urls = [result["url"] for result in results if "url" in result]
                return urls
            elif task_status == "FAILED":
                code = output.get("code", "UnknownError")
                message = output.get("message", "无错误消息")
                raise JobFailedException(code, message)
            elif task_status == "RUNNING":
                if not wait:
                    return None
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise APIException("等待作业完成超时。")
                time.sleep(poll_interval)
            else:
                raise APIException(f"未知的任务状态: {task_status}")

    def call(self, prompt: str, poll_interval: int = 3, timeout: int = 300) -> List[str]:
        task_id = self.submit_text2image_job(prompt=prompt)
        urls = self.get_job_result(task_id, wait=True, poll_interval=poll_interval, timeout=timeout)
        if urls is None:
            raise APIException("作业未完成且未选择等待。")
        return urls
