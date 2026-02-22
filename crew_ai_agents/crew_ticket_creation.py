from crewai.flow.flow import Flow, listen, or_, start, and_,router
from pydantic import BaseModel
import random


class TicketState(BaseModel):
    priority: str = "low"

class supportflow(Flow):
    @start()
    def live_chat_support(self):
        return "Support requests raised through live chat"
    
    @start()
    def email_support(self):
        return "Support requests raised through email"
    
    @listen(or_(live_chat_support,email_support))
    def log_request(self, request_source):
        return (f"Request raised from {request_source}")


    

class TicketEscalationFlow(Flow):

    @start()
    def user_confirms_issue(self):
        self.state["user_confirmation"] = False
        print("User confirmed they still need assistance.")

    @listen(user_confirms_issue)
    def agent_reviews_ticket(self):
        self.state["agent_review"] = False
        print("Support agent has reviewed the ticket.")

    @listen(and_(user_confirms_issue, agent_reviews_ticket))
    def escalate_ticket(self):
        print("Escalating ticket to Level 2 support!")
        

class TicketRoutingFlow(Flow[TicketState]):
    @start()
    def classify_ticket(self):
        print("Classifying ticket...")
        self.state.priority = random.choice(["low", "high"])
        print(f"Ticket classified as {self.state.priority}")
        
    @router(classify_ticket)
    def route_ticket(self):
        if self.state.priority == "high":
            return "chat_support"
        if self.state.priority == "low":
            return "email_support"
        
    @listen("chat_support")
    def assign_to_chat_agent(self):
        print("Assigning ticket to chat agent...")
        
    @listen("email_support")
    def send_email(self):
        print("Sending to email response queue")

        

    
    



async def main():
    flow = TicketRoutingFlow()
    final_result = await flow.kickoff_async()
    print(final_result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())