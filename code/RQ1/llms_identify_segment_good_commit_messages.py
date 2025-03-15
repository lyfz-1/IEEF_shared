import json
import time
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import threading

# API_KEY and BASE_URL
API_KEY = ""
BASE_URL = ""

requests_per_second = 33
lock = threading.Lock()

last_request_time = [0]

def get_completion(prompt, model="gpt-4o-mini"):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    with lock:
        current_time = time.time()
        time_since_last_request = current_time - last_request_time[0]

        if time_since_last_request < 1 / requests_per_second:
            time.sleep((1 / requests_per_second) - time_since_last_request)
        last_request_time[0] = time.time()

    response = client.chat.completions.create(
      model=model,
      messages=[
          {
              "role": "user",
              "content": prompt
          }
      ]
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content

def cleanResponse(response):
    cleaned_text = re.sub(r'\n+', '\n', response)
    if len(cleaned_text.split("\n")) > 3:
        temp_text = cleaned_text.split("\n")[1:]
        cleaned_text = "\n".join(temp_text)
    if "reason" in cleaned_text and len(cleaned_text.split("\n")) == 3:
        temp_text = cleaned_text.split("\n")[1:]
        cleaned_text = "\n".join(temp_text)
    return cleaned_text

def process_message(idx, item):
    diff, msg, commit_time = item
    test_diff = str(diff).strip()
    test_msg = str(msg).strip()

    prompt = "You are now an expert in natural language semantic analysis. Commit messages are concise summaries of code changes. A good commit message must contain elements of both: (1) what: describes the content or the edit operation of the code changes; (2) why: describes the reason, purpose, or impact of the code changes.\n" \
             "Now, please determine if a commit message is a good commit message. If it's a good commit message (that is, it contains both what and why information), answer with the number 1 and separate what and why parts from the commit message itself. If it's not a good commit message, answer with the number 0 and point out the missing information (miss what, miss why, miss both). Note that you do not need to generate the missing part.\n" \
             "Here are four examples to help you understand (please obey the result format of the following Examples, and do not return json string): \n" \
             "Example 1:\ncommit message: removed explicit toString ( ) call to prevent NPE when userInfo is null\nanswer: 1\nwhat: removed explicit toString ( ) call\nwhy: prevent NPE when userInfo is null\n" \
             "Example 2:\ncommit message: add GitHub icon to toolbar\nanswer: 0\nreason: miss why\n" \
             "Example 3:\ncommit message: Fix test failures for CASSANDRA\nanswer: 0\nreason: miss what\n" \
             "Example 4:\ncommit message: demos\nanswer: 0\nreason: miss both\n" \
             f"Test commit message: {test_msg}"

    result = get_completion(prompt)
    result = result.strip()
    result = cleanResponse(result)
    result_parts = result.split("\n")

    if len(result_parts) == 2:
        # not good commit message
        if "what" in result:
            json_data = {"commit_time": commit_time, "diff": test_diff, "commit_message": test_msg, "predict_label": 0,
                         "reason": "miss what"}
        elif "why" in result:
            json_data = {"commit_time": commit_time, "diff": test_diff, "commit_message": test_msg, "predict_label": 0,
                         "reason": "miss why"}
        else:
            json_data = {"commit_time": commit_time, "diff": test_diff, "commit_message": test_msg, "predict_label": 0,
                         "reason": "miss both"}
    elif len(result_parts) == 3:
        # good commit message
        if len(result_parts[1].split("what: ")) == 2 or len(result_parts[1].split("What: ")) == 2:
            what_info = result_parts[1].split("what: ")[1].strip() if "what" in result_parts[1] else \
            result_parts[1].split("What: ")[1].strip()
        elif result_parts[1] == "":
            test_msg_tokens = test_msg.split(" ")
            test_msg_tokens_len = len(test_msg_tokens)
            what_info = " ".join(test_msg_tokens[:test_msg_tokens_len // 2]).strip()
        else:
            what_info = result_parts[1].strip()

        if len(result_parts[2].split("why: ")) == 2 or len(result_parts[2].split("Why: ")) == 2:
            why_info = result_parts[2].split("why: ")[1].strip() if "why" in result_parts[2] else \
            result_parts[2].split("Why: ")[1].strip()
        elif result_parts[2] == "":
            test_msg_tokens = test_msg.split(" ")
            test_msg_tokens_len = len(test_msg_tokens)
            why_info = " ".join(test_msg_tokens[test_msg_tokens_len // 2:]).strip()
        else:
            why_info = result_parts[2].strip()

        json_data = {"commit_time": commit_time, "diff": test_diff, "commit_message": test_msg, "predict_label": 1,
                     "what": what_info, "why": why_info}
    else:
        # other cases
        json_data = {"commit_time": commit_time, "diff": test_diff, "commit_message": test_msg, "predict_label": 0,
                     "reason": "LLM failed to identify and segment the commit message"}
    return idx, json_data

if __name__ == '__main__':
    s_time = time.time()

    data = pd.read_csv("")
    diffs = data["diff"].tolist()
    msgs = data["commit_message"].tolist()
    commit_times = data["commit_time"].tolist()

    json_list = [None] * len(msgs)
    counter = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_message, idx, item) for idx, item in enumerate(zip(diffs, msgs, commit_times))]

        for future in as_completed(futures):
            idx, result = future.result()
            json_list[idx] = result
            counter += 1
            if counter % 10000 == 0:
                with open("", "w",
                          encoding="utf-8") as f:
                    json.dump(json_list, f, ensure_ascii=False, indent=4)

    e_time = time.time()
    print(f'Total time cost: {e_time - s_time} s')
