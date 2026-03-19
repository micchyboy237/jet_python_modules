import asyncio

import dspy
from jet.libs.dspy.custom_config import configure_dspy_lm

configure_dspy_lm()

predict = dspy.Predict("question->answer")

stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)


async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

    async for chunk in output_stream:
        print(chunk)


asyncio.run(read_output_stream())
