import os
import asyncio
import json
from typing import List, Dict, Any
from openai import AsyncOpenAI
import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("HF_TOKEN", ""))
ENV_URL = os.environ.get("ENV_URL", "http://127.0.0.1:7860")

def log_start(task: str, env: str, model: str):
    # Strictly following: [START] task=<task_name> env=<benchmark> model=<model_name>
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    # Strictly following: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    done_str = str(done).lower()
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    # Strictly following: [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
    success_str = str(success).lower()
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

async def get_model_action(client: AsyncOpenAI, step: int, obs: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    system_prompt = """You are an excellent customer support triage agent. 
You will be given the current state of tickets and must choose ONE action from:
1. {"action_type": "assign", "ticket_id": "...", "department": "billing|sales|tech_support|returns"}
2. {"action_type": "request_info", "ticket_id": "...", "message": "..."}
3. {"action_type": "close", "ticket_id": "...", "reason": "spam|resolved"}

Respond ONLY with valid JSON.
"""
    history_str = "\n".join(history)
    user_prompt = f"Observation: {json.dumps(obs)}\nHistory:\n{history_str}\nWhat is your next action JSON?"
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    
    content = response.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content.replace("```json\n", "").replace("\n```", "")
    elif content.startswith("```"):
        content = content.replace("```\n", "").replace("\n```", "")
        
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback action
        return {"action_type": "close", "ticket_id": "t1", "reason": "parsing_error"}

async def run_task(task_id: int):
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    log_start(task=f"Task {task_id}", env="CustomerSupportTriage", model=MODEL_NAME)
    
    async with httpx.AsyncClient() as http:
        # Reset environment for this task
        resp = await http.post(f"{ENV_URL}/reset", json={"episode_id": str(task_id)})
        obs = resp.json()
        
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        success = False
        
        MAX_STEPS = 10
        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break
                
            action_dict = await get_model_action(client, step, obs, history)
            action_str = json.dumps(action_dict)
            history.append(f"Agent Action: {action_str}")
            
            resp = await http.post(f"{ENV_URL}/step", json={"action": action_dict})
            if resp.status_code != 200:
                log_step(step=step, action=action_str, reward=0.0, done=True, error=resp.text)
                break
                
            obs = resp.json()
            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)
            
            rewards.append(reward)
            steps_taken = step
            
            history.append(f"Env Reply: {obs.get('agent_message')} | Reward: {reward}")
            
            log_step(step=step, action=action_str, reward=reward, done=done)
            
            if done:
                break
                
        score = sum(rewards)
        # Cap score between 0 and 1.0 roughly or use the final grade if provided
        final_grade = obs.get("metadata", {}).get("final_score", score)
        success = final_grade >= 0.8
        
        log_end(success=success, steps=steps_taken, rewards=rewards)

async def main():
    for i in range(3):
        await run_task(i)

if __name__ == "__main__":
    asyncio.run(main())
