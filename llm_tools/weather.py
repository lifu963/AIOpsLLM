from typing import Dict, Optional, Union, Tuple
import requests

import pandas as pd

from agent.schema import ResponseStatus
from llm_tools import BaseTool


class WeatherTool(BaseTool):
    name = 'weather'
    description = '获取对应城市的天气数据'
    parameters = [{
        'name': 'location',
        'type': 'string',
        'description': '城市/区具体名称，如`北京市海淀区`请描述为`海淀区`,`福州`请描述为`福州市`',
        'required': True
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)

        self.url = 'https://restapi.amap.com/v3/weather/weatherInfo?city={city}&key={key}'
        self.city_df = pd.read_excel('llm_tools/resources/AMap_adcode_citycode.xlsx')

        self.token = self.cfg.get('token', 'de8238e5337d39b3a9fe74a218508d57')
        assert self.token != '', 'weather api token must be acquired through ' \
            'https://lbs.amap.com/api/webservice/guide/create-project/get-key and set by AMAP_TOKEN'

    def get_city_adcode(self, city_name):
        filtered_df = self.city_df[self.city_df['中文名'] == city_name]
        if len(filtered_df['adcode'].values) == 0:
            raise ValueError(f'location {city_name} not found, availables are {self.city_df["中文名"]}')
        else:
            return filtered_df['adcode'].values[0]

    def call(self, params: Union[str, dict], **kwargs) -> Tuple:
        params = self._verify_json_format_args(params)

        location = params['location']
        response = requests.get(self.url.format(city=self.get_city_adcode(location), key=self.token))
        data = response.json()
        if data['status'] == '0':
            raise RuntimeError(data)
        else:
            weather = data['lives'][0]['weather']
            temperature = data['lives'][0]['temperature']
            humidity = data['lives'][0]['humidity']
            return ResponseStatus.STR, f'{location}的天气是{weather}，温度是{temperature}度，湿度是{humidity}%'
