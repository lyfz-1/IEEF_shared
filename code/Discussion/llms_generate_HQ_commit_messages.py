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

requests_per_second = 15
lock = threading.Lock()

def get_completion(prompt, model="claude-3.5-sonnet"):

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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

def process_message(idx, diff):
    test_diff = diff

    example_diff_1 = "mmm a / portlet / src / main / java / org / springframework / security / ui / portlet / PortletAuthenticationDetails . java <nl> ppp b / portlet / src / main / java / org / springframework / security / ui / portlet / PortletAuthenticationDetails . java <nl> public class PortletAuthenticationDetails implements Serializable { <nl> } <nl> <nl> public String toString ( ) { <nl> - return ' User info : ' + userInfo . toString ( ) ; <nl> + return ' User info : ' + userInfo ; <nl> } <nl> }"
    example_msg_1 = "Removed explicit toString ( ) call to prevent NPE when userInfo is null"

    example_diff_2 = "mmm UserInterfaceComponent . java <nl> ppp UserInterfaceComponent . java <nl> - import com . speedment . internal . ui . util . LogUtil ; <nl> + import com . speedment . internal . ui . util . OutputUtil ; <nl> mmm UISession . java <nl> ppp UISession . java <nl> - import static com . speedment . internal . ui . util . LogUtil . error ; <nl> - import static com . speedment . internal . ui . util . LogUtil . info ; <nl> - import static com . speedment . internal . ui . util . LogUtil . success ; <nl> + import static com . speedment . internal . ui . util . OutputUtil . error ; <nl> + import static com . speedment . internal . ui . util . OutputUtil . info ; <nl> + import static com . speedment . internal . ui . util . OutputUtil . success ; <nl> + import static com . speedment . internal . util . TextUtil . alignRight ; <nl> + import static java . util . Objects . requireNonNull ; <nl> mmm LogUtil . java <nl> ppp LogUtil . java <nl> - public final class LogUtil { <nl> + public final class OutputUtil { <nl> - private LogUtil() { <nl> + private OutputUtil() { <nl>"
    example_msg_2 = "Change name of logutil to outpututil to prevent confusion with logger"

    example_diff_3 = "mmm PinotTaskManagerStatelessTest . java <nl> ppp PinotTaskManagerStatelessTest . java <nl> + import org . slf4j . Logger ; <nl> + import org . slf4j . LoggerFactory ; <nl> + private static final Logger LOGGER = LoggerFactory . getLogger(PinotTaskManagerStatelessTest . class) ; <nl> - List<? extends Trigger> triggersOfJob = scheduler . getTriggersOfJob(jobKey) ; <nl> - Trigger trigger = triggersOfJob . iterator() . next() ; <nl> - assertTrue(trigger instanceof CronTrigger) ; <nl> - assertEquals(((CronTrigger) trigger) . getCronExpression(), cronExpression) ; <nl> + TestUtils . waitForCondition(aVoid -> { <nl> + try { <nl> + List<? extends Trigger> triggersOfJob = scheduler . getTriggersOfJob(jobKey) ; <nl> + Trigger trigger = triggersOfJob . iterator() . next() ; <nl> + assertTrue(trigger instanceof CronTrigger) ; <nl> + assertEquals(((CronTrigger) trigger) . getCronExpression(), cronExpression) ; <nl> + } catch (SchedulerException ex) { <nl> + throw new RuntimeException(ex) ; <nl> + } catch (AssertionError assertionError) { <nl> + LOGGER . warn(\"Unexpected cron expression .  Hasn't been replicated yet?\", assertionError) ; <nl> + } <nl> + return true ; <nl> + }, TIMEOUT_IN_MS, 500L, \"Cron expression didn't change to \" cronExpression) ; <nl>"
    example_msg_3 = "Retry validatejobs for a while to avoid flaky error ( # 8999 )"

    # prompt_0_shot = "You are an experienced Java developer. A commit message is a concise summary of code changes. A good commit message must contain elements of both: (1) what: describes the content or the edit operation of the code changes; (2) why: describes the reason, purpose, or impact of the code changes.\n" \
    #          "Now, please help to write a good commit message for the following code changes:\n" \
    #          f"{test_diff}\n" \
    #          f"Note that your answer only needs to contain one line of text (the commit message), not json or anything else."

    # prompt_1_shot = "You are an experienced Java developer. A commit message is a concise summary of code changes. A good commit message must contain elements of both: (1) what: describes the content or the edit operation of the code changes; (2) why: describes the reason, purpose, or impact of the code changes.\n" \
    #                 "Here is an example for you:\n" \
    #                 "code changes:\n" \
    #                 f"{example_diff_1}\n" \
    #                 "commit message:\n" \
    #                 f"{example_msg_1}\n" \
    #          "Now, please help to write a good commit message for the following code changes:\n" \
    #          f"{test_diff}\n" \
    #          f"Note that your answer only needs to contain one line of text (the commit message), not json or anything else."


    prompt_3_shot = "You are an experienced Java developer. A commit message is a concise summary of code changes. A good commit message must contain elements of both: (1) what: describes the content or the edit operation of the code changes; (2) why: describes the reason, purpose, or impact of the code changes.\n" \
                    "Here are three examples for you:\n" \
                    "Example 1:\n" \
                    "code changes:\n" \
                    f"{example_diff_1}\n" \
                    "commit message:\n" \
                    f"{example_msg_1}\n" \
                    "Example 2:\n" \
                    "code changes:\n" \
                    f"{example_diff_2}\n" \
                    "commit message:\n" \
                    f"{example_msg_2}\n" \
                    "Example 3:\n" \
                    "code changes:\n" \
                    f"{example_diff_3}\n" \
                    "commit message:\n" \
                    f"{example_msg_3}\n" \
             "Now, please help to write a good commit message for the following code changes:\n" \
             f"{test_diff}\n"

    with lock:
        time.sleep(1 / requests_per_second)

    result = get_completion(prompt_3_shot)
    result = result.strip()
    result = cleanResponse(result)

    json_data = {
        "diff": test_diff,
        "msg_gen": result
    }

    return idx, json_data

if __name__ == '__main__':
    s_time = time.time()

    diff_list = []

    with open("data_stage3_test.jsonl", "r", encoding="utf-8") as fr:
        for line in fr:
            json_obj = json.loads(line)
            diff_text = " ".join(json_obj["diff"][1:-1])
            diff_list.append(diff_text)

    diff_list = diff_list[:1000]

    json_list = [None] * len(diff_list)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_message, idx, diff) for idx, diff in enumerate(diff_list)]

        for future in as_completed(futures):
            idx, result = future.result()
            json_list[idx] = result

        with open(f"results/test_claude-3.5-sonnet_3-shot_results.json", "w",
                  encoding="utf-8") as f2:
            json.dump(json_list, f2, ensure_ascii=False, indent=4)

    print("LLM generate good commit message done!")
    e_time = time.time()
    print(f'Time cost: {e_time - s_time} s')