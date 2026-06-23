import requests
import time

res = requests.post("http://127.0.0.1:8000/api/generate-full", json={
    "concept": "Rektorat ITS wisuda",
    "use_rag": True,
    "prompt_technique": "zero-shot"
})
task_id = res.json()["task_id"]
print("Task ID:", task_id)

for _ in range(30):
    time.sleep(2)
    res = requests.get(f"http://127.0.0.1:8000/api/task/{task_id}")
    status = res.json()["status"]
    print("Status:", status)
    if status in ["completed", "failed"]:
        print(res.json())
        break
