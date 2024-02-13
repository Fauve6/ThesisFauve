import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
# openai.File.create(
#   file=open("data/train/train_everything.jsonl", "rb"),
#   purpose='fine-tune'
# )

openai.FineTuningJob.create(training_file="", model="gpt-3.5-turbo-1106", suffix='english_1106')

