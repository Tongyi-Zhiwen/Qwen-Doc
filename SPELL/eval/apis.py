from openai import OpenAI
import time
def query_local_vllm(messages: list, args):
    client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
        timeout=1800
    )
    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_output_len,
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    print("Max tries. Failed.")
    return ''
    