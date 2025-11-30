import asyncio
from tqdm.auto import tqdm
import traceback
import json


async def map_progress(seq, f, max_concurrency=6):
    """Asynchronously map async function f over seq with progress bar."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(el):
        async with semaphore:
            return await f(el)

    # create one coroutine per element
    coros = [run(el) for el in seq]

    # turn them into tasks that complete as they finish
    completed = asyncio.as_completed(coros)

    results = []

    for coro in tqdm(completed, total=len(seq)):
        result = await coro
        results.append(result)

    return results


def simplify_messages(messages):
    messages_simplified = []

    for m in messages:
        parts = []

        for original_part in m.parts:
            kind = original_part.part_kind
            # print(original_part)
            part = {"kind": kind}
            if kind == "user-prompt":
                part["content"] = original_part.content
            if kind == "tool-call":
                if original_part.tool_name == "final_result":
                    continue

                part["tool_name"] = original_part.tool_name
                part["args"] = json.loads(original_part.args)

            if kind == "tool-return":
                # continue
                if original_part.tool_name == "search_es":
                    # print(original_part.content)
                    keys_to_keep = ["start_time", "video_id", "content"]
                    filtered = [
                        {k: d[k] for k in keys_to_keep if k in d}
                        for d in original_part.content
                    ]
                    part["content"] = filtered
                else:
                    continue

            if kind == "text":
                part["content"] = original_part.content

            parts.append(part)

        if len(parts) > 0:
            messages_simplified.extend(parts)

    return messages_simplified


def process_result(q, result):
    row = {}

    row["question"] = q["user_question"]
    row["answer"] = result.output.format_youtube_summary()
    row["product_answer"] = [product.name for product in result.output.products]
    row["product_reason_answer"] = [
        product.reason for product in result.output.products
    ]
    row["messages"] = simplify_messages(result.new_messages())

    # row['num_tool_calls'] = count_tool_calls(row['messages'])

    row["original_question"] = q
    row["original_result"] = result

    return row
