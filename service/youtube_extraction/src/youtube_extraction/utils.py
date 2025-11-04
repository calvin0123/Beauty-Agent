from pydantic import BaseModel, Field
from typing import List

from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()
openai_client = OpenAI()


class VideoCategory(Enum):
    makeup = "makeup"
    skincare = "skincare"
    beauty = "beauty"


class YTSummaryResponse(BaseModel):
    summary_zh: str = Field(description="The summary of the youtube video transcript.")
    title_zh: str = Field(description="The title of the youtube video title")
    summary_en: str = Field(description="English translation of the summary.")
    title_en: str = Field(description="English translation of the title.")
    category: VideoCategory = Field(description=(
        "Main beauty category of the video. Choose one of:\n"
        "- makeup: tutorials, reviews, product try-ons\n"
        "- skincare: routines, product reviews, ingredients\n"
        "- haircare: styling, coloring, or hair treatment tips"
    ))



def youtuber_summarize(user_prompt, output_format, model="gpt-4o-mini"):
    
    instructions = """
    You are a multilingual beauty content analyst.
    Summarize YouTube transcripts about beauty products in Chinese,
    then provide English translations for the summary and title.
    Assign a category: makeup, skincare, or haircare.
    """.strip()

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = openai_client.responses.parse(
        model=model,
        input=messages,
        text_format=output_format
    )

    return response.output[0].content[0].parsed


def translate_english(text):

    from deep_translator import GoogleTranslator

    proxies_example = {
        "http": os.environ['WEBSHARE_HTTP'],
        "https": os.environ['WEBSHARE_HTTPS']
    }

    translated = GoogleTranslator(source='zh-TW', target='en', proxies=proxies_example).translate(text)
    return translated
