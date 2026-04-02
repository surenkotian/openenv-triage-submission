---
title: Customer Support Triage
emoji: 🎟️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags: [openenv, reinforcement-learning, text-classification, multi-turn]
---

# Customer Support Triage - OpenEnv

The `CustomerSupportTriage` OpenEnv environment places the agent in the shoes of a customer support team member managing incoming support tickets. This models a very common and highly valuable real-world business task.

## Key Features

- **Multi-turn interactiveness**: Agents don't just assign tickets, they may need to ask for missing information and await a customer reply in a subsequent step.
- **Progressive Difficulty**: Tasks scale from a simple single-ticket triage to managing a queue of disparate requests with missing data.
- **Fully Typed Interface**: Strictly implemented with Pydantic typing for observation, actions, and step states.

## Observation Space

At each step, the model observes the current state of tickets and any messages from the system.
```json
{
  "open_tickets": [
    {
      "id": "t1",
      "subject": "Billing issue",
      "body": "My account is locked."
    }
  ],
  "agent_message": "Ticket assigned successfully.",
  "reward": 0.0,
  "done": false
}
```

## Action Space

The agent responds with a structured JSON action:
```json
{
  "action_type": "assign | request_info | close",
  "ticket_id": "t1",
  "department": "billing", 
  "message": "Can you provide your order ID?", 
  "reason": "spam" 
}
```

## Tasks Sequence

- **Task 0 (Easy)**: A single straightforward billing ticket to be assigned.
- **Task 1 (Medium)**: Three distinct incoming tickets that must be correctly mapped to tech_support or sales, and closing obvious spam.
- **Task 2 (Hard)**: The agent has incomplete information on a refund ticket. They must issue a `request_info` action to get the order ID. The environment simulates the customer response in the next tick, after which the agent must classify the ticket.

## Running the Environment

Build and run via Docker:
```bash
docker build -t openenv-submission .
docker run -p 7860:7860 openenv-submission
```

## Baseline Evaluation

We provide `inference.py` which interfaces via the OpenAI client, hitting our deployed endpoints sequentially using prompt structuring. 

Run inference:
```bash
python inference.py
```

### Baseline Performance
- Task 0: 1.0 (Optimal)
- Task 1: 1.0 (Optimal)
- Task 2: ~0.8-1.0 (Dependent on model multi-turn tracking)
