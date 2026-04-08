import os
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from openenv.core import Environment, Observation, Action, State, create_fastapi_app

class Ticket(BaseModel):
    id: str
    subject: str
    body: str

class TriageObservation(Observation):
    open_tickets: List[Ticket]
    agent_message: Optional[str] = None
    
class TriageAction(Action):
    action_type: Literal["assign", "request_info", "close"]
    ticket_id: str
    department: Optional[str] = None
    message: Optional[str] = None
    reason: Optional[str] = None

class TriageState(State):
    open_tickets: List[Ticket]
    assigned_tickets: Dict[str, str] = Field(default_factory=dict)
    closed_tickets: Dict[str, str] = Field(default_factory=dict)
    current_task: int = 0
    steps: int = 0
    task_started: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CustomerSupportEnv(Environment[TriageAction, TriageObservation, TriageState]):
    _global_state = TriageState(open_tickets=[])

    def __init__(self):
        super().__init__()
        self._state = CustomerSupportEnv._global_state
        self.num_tasks = 3
        
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> TriageObservation:
        CustomerSupportEnv._global_state = TriageState(open_tickets=[])
        self._state = CustomerSupportEnv._global_state
        
        # Load task based on metadata or random
        task = 0
        if episode_id is not None:
            try:
                task = int(episode_id)
            except ValueError:
                pass
        if "task" in kwargs:
            try:
                task = int(kwargs["task"])
            except ValueError:
                pass
                
        self._state.current_task = max(0, min(task, self.num_tasks - 1))
        self._state.task_started = True
        
        self._setup_task(self._state.current_task)
        
        return TriageObservation(
            open_tickets=self._state.open_tickets,
            agent_message=f"Agent starting task {self._state.current_task}",
            done=False,
            reward=0.01
        )
        
    def _setup_task(self, task: int):
        if task == 0:
            self._state.open_tickets = [
                Ticket(id="t1", subject="Billing issue", body="My account is locked.")
            ]
        elif task == 1:
            self._state.open_tickets = [
                Ticket(id="t1", subject="Upgrade plan", body="How do I change my subscription?"),
                Ticket(id="t2", subject="URGENT!!!", body="Buy cheap pills now!"),
                Ticket(id="t3", subject="App crashes", body="Crashing on launch.")
            ]
        elif task == 2:
            self._state.open_tickets = [
                Ticket(id="t1", subject="Refund Request", body="I need a refund."),
                Ticket(id="t2", subject="Verification", body="Email not verifying.")
            ]
            self._state.metadata["pending_reply"] = False

    def step(self, action: TriageAction, timeout_s: Optional[float] = None, **kwargs: Any) -> TriageObservation:
        self._state.steps += 1
        reward = 0.01
        message = ""
        done = False
        
        # Validate ticket exists
        ticket = next((t for t in self._state.open_tickets if t.id == action.ticket_id), None)
        
        if ticket is None:
            message = "Error: Invalid ticket ID."
            reward -= 0.1
        else:
            if action.action_type == "assign":
                if not action.department:
                    message = "Error: assign requires a department."
                    reward -= 0.1
                else:
                    self._state.assigned_tickets[ticket.id] = action.department
                    self._state.open_tickets.remove(ticket)
                    message = f"Ticket {ticket.id} assigned to {action.department}."
                    reward += 0.2
            elif action.action_type == "close":
                self._state.closed_tickets[ticket.id] = action.reason or "none"
                self._state.open_tickets.remove(ticket)
                message = f"Ticket {ticket.id} closed."
                reward += 0.1
            elif action.action_type == "request_info":
                if not action.message:
                    message = "Error: request_info requires a message."
                    reward -= 0.1
                else:
                    message = f"Requested info for {ticket.id}."
                    # specifically for task 2 triggers reply
                    if self._state.current_task == 2 and ticket.id == "t1":
                        self._state.metadata["pending_reply"] = True
                        reward += 0.5
                    else:
                        reward -= 0.1 # Not needed here

        # Task 2 dynamic update
        if self._state.current_task == 2 and self._state.metadata.get("pending_reply"):
            ticket = next((t for t in self._state.open_tickets if t.id == "t1"), None)
            if ticket:
                ticket.body += "\n[Customer Reply]: My Order ID is 12345."
                self._state.metadata["pending_reply"] = False
                message += " Customer replied to t1."
        
        # We use very small step rewards to keep the cumulative total within (0, 1)
        # while ensuring no single step reward is ever exactly 0.0
        step_reward = 0.01
        
        # Checking done condition
        if len(self._state.open_tickets) == 0 or self._state.steps >= 10:
            done = True
            # The final reward is the primary score
            final_grade = self._grade_task()
            
            # Clamp the final grade to ensure it's strictly within (0.01, 0.99)
            # regardless of any cumulative additions that might have happened
            step_reward = final_grade

        return TriageObservation(
            open_tickets=self._state.open_tickets,
            agent_message=message,
            reward=max(0.01, min(0.99, step_reward)),
            done=done,
            metadata={"final_score": self._grade_task()}
        )
        
    def _grade_task(self) -> float:
        task = self._state.current_task
        score = 0.0
        if task == 0:
            if self._state.assigned_tickets.get("t1") == "billing":
                score = 1.0
        elif task == 1:
            if self._state.assigned_tickets.get("t1") == "sales": score += 0.33
            if "t2" in self._state.closed_tickets: score += 0.33
            if self._state.assigned_tickets.get("t3") == "tech_support": score += 0.34
        elif task == 2:
            if self._state.assigned_tickets.get("t1") in ["billing", "returns", "sales"]:
                score += 0.5 # only if they assigned it
            if self._state.assigned_tickets.get("t2") == "tech_support":
                score += 0.5
        
        # STRICTLY between 0 and 1 as per Phase 2 requirements (0.01 to 0.99)
        return max(0.01, min(0.99, score))

    @property
    def state(self) -> TriageState:
        return self._state

# Create the FastAPI app that OpenEnv runner will use
app = create_fastapi_app(CustomerSupportEnv, TriageAction, TriageObservation)

# Add a welcome route so the browser doesn't show a 404 when people visit your Space!
@app.get("/")
async def root():
    return {
        "environment": "Customer Support Triage",
        "status": "online",
        "message": "OpenEnv endpoints (/reset, /step, /metadata) are ready."
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

# If run directly (for debugging)
if __name__ == "__main__":
    main()
