import dspy
import os

lm = dspy.LM(model="openai/gpt-oss-20b",
    api_key=os.getenv("LIGHTNING_API_KEY"),
    api_base="https://lightning.ai/api/v1/")
dspy.configure(lm=lm)