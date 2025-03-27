import os
from typing import Optional, List
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = ["*"]
        
        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0
        
        # Custom configuration for this filter
        log_level: str = "info"
        include_timestamps: bool = True

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # This is a minimal filter for testing purposes
        self.type = "filter"
        self.name = "Simple Echo Filter"

        # Initialize with default configuration
        self.valves = self.Valves()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"Simple Echo Filter starting up with log level: {self.valves.log_level}")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print("Simple Echo Filter shutting down")

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"Valves updated: log_level = {self.valves.log_level}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Simple echo filter that logs the incoming message and passes it through unchanged.
        
        Args:
            body: The request body containing the messages
            user: Optional user information
            
        Returns:
            The original body without modifications
        """
        print(f"Echo Filter received message:")
        
        if "messages" in body and len(body["messages"]) > 0:
            user_message = body["messages"][-1]["content"]
            print(f"User message: {user_message}")
            
            if user and "username" in user:
                print(f"From user: {user['username']}")
        
        # Simply pass through the message without modification
        return body