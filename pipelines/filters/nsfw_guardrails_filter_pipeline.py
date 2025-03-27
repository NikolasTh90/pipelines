import os
import sys
import subprocess
import importlib.util
import time
from typing import Optional, List
from pydantic import BaseModel
import re

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to
        pipelines: List[str] = ["*"]
        
        # Assign a priority level to the filter pipeline
        priority: int = 0
        
        # Custom configuration for NSFW detection
        threshold: float = 0.8  # Threshold for NSFW detection (0.0 to 1.0)
        log_level: str = "info"
        block_nsfw_content: bool = True
        
        # Basic NSFW word list fallback when Guardrails isn't available
        use_fallback_detection: bool = True
        
        # Installation options
        attempt_auto_install: bool = True
        
    def __init__(self):
        # This filter uses Guardrails AI's NSFWText module to detect and block NSFW content
        self.type = "filter"
        self.name = "Guardrails NSFW Filter"

        # Initialize with default configuration
        self.valves = self.Valves()
        
        # Try to import Guardrails components
        self.guardrails_mode = self._setup_guardrails()
        
        # Basic regex pattern for NSFW content detection as fallback
        self.nsfw_pattern = None
        if self.guardrails_mode == "fallback" and self.valves.use_fallback_detection:
            # A very basic list of NSFW terms (expand as needed)
            nsfw_terms = [
                'explicit', 'nsfw', 'xxx', 'porn', 'sexual', 'obscene'
                # Add more terms as needed
            ]
            pattern = r'\b(' + '|'.join(nsfw_terms) + r')\b'
            self.nsfw_pattern = re.compile(pattern, re.IGNORECASE)

    def _setup_guardrails(self):
        """Set up Guardrails and return the mode (guardrails, fallback, or disabled)"""
        nsfw_validator_available = False
        
        # First try to import Guardrails
        try:
            import guardrails as gd
            from guardrails.hub import NSFWText
            nsfw_validator_available = True
        except ImportError as e:
            print(f"Guardrails import error: {str(e)}")
            # If we can't import guardrails or NSFWText is not in hub, try to install
            if self.valves.attempt_auto_install:
                print("Attempting to install Guardrails and NSFWText validator...")
                installed = self._install_guardrails()
                if installed:
                    try:
                        # Re-import after installation
                        import guardrails as gd
                        from guardrails.hub import NSFWText
                        nsfw_validator_available = True
                    except ImportError as e2:
                        print(f"Still could not import after installation: {str(e2)}")
            
        # If we have guardrails, try to initialize the validator
        if nsfw_validator_available:
            try:
                from guardrails.hub import NSFWText
                self.nsfw_validator = NSFWText(threshold=self.valves.threshold)
                print(f"Successfully initialized Guardrails NSFWText validator with threshold: {self.valves.threshold}")
                return "guardrails"
            except Exception as e:
                print(f"Error initializing Guardrails NSFWText validator: {str(e)}")
                
        # Fall back to basic detection
        print("Using basic fallback NSFW detection.")
        return "fallback"
                
    def _install_guardrails(self):
        """Attempt to install Guardrails and NSFWText using subprocess"""
        try:
            # First install the guardrails-ai package
            print("Installing guardrails-ai package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "guardrails-ai"])
            
            # Wait a moment for installation to complete
            time.sleep(2)
            
            # Then install the NSFWText validator from hub
            print("Installing NSFWText validator from Guardrails hub...")
            subprocess.check_call([sys.executable, "-m", "guardrails", "hub", "install", "hub://guardrails/nsfw_text"])
            
            print("Installation completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Installation failed: {str(e)}")
            print("To install manually, run:")
            print("  pip install guardrails-ai")
            print("  guardrails hub install hub://guardrails/nsfw_text")
            return False
        except Exception as e:
            print(f"Unexpected error during installation: {str(e)}")
            return False

    async def on_startup(self):
        # This function is called when the server is started
        print(f"Guardrails NSFW Filter starting up")
        
        if self.guardrails_mode == "guardrails":
            print(f"Running with Guardrails NSFWText validator")
            print(f"NSFW detection threshold: {self.valves.threshold}")
        elif self.guardrails_mode == "fallback" and self.valves.use_fallback_detection:
            print("Running in fallback mode with basic pattern matching")
            print("For better detection, install the full Guardrails NSFWText validator:")
            print("  pip install guardrails-ai")
            print("  guardrails hub install hub://guardrails/nsfw_text")
        else:
            print("NSFW detection disabled")
            
        print(f"Block NSFW content: {self.valves.block_nsfw_content}")

    async def on_shutdown(self):
        # This function is called when the server is stopped
        print("Guardrails NSFW Filter shutting down")

    async def on_valves_updated(self):
        # This function is called when the valves are updated
        print(f"Valves updated: NSFW threshold = {self.valves.threshold}")
        print(f"Block NSFW content: {self.valves.block_nsfw_content}")
        
        # Update the validator if we're using Guardrails
        if self.guardrails_mode == "guardrails":
            try:
                from guardrails.hub import NSFWText
                self.nsfw_validator = NSFWText(threshold=self.valves.threshold)
                print(f"Updated Guardrails NSFW validator with new threshold: {self.valves.threshold}")
            except Exception as e:
                print(f"Error updating Guardrails NSFW validator: {str(e)}")
        
        # Update regex pattern if we're using fallback
        if self.guardrails_mode == "fallback" and self.valves.use_fallback_detection:
            nsfw_terms = [
                'explicit', 'nsfw', 'xxx', 'porn', 'sexual', 'obscene'
                # Add more terms as needed
            ]
            pattern = r'\b(' + '|'.join(nsfw_terms) + r')\b'
            self.nsfw_pattern = re.compile(pattern, re.IGNORECASE)

    def _check_content_fallback(self, text):
        """Fallback NSFW detection using regex patterns"""
        if not self.nsfw_pattern:
            return False, "NSFW detection disabled"
            
        if self.nsfw_pattern.search(text):
            return True, "Potentially NSFW content detected based on keyword matching"
        return False, ""

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Filter incoming messages for NSFW content before they reach the LLM.
        
        Args:
            body: The request body containing the messages
            user: Optional user information
            
        Returns:
            The original body if the content is safe
            
        Raises:
            Exception: If NSFW content is detected and blocking is enabled
        """
        if "messages" not in body or not body["messages"]:
            return body
            
        user_message = body["messages"][-1]["content"]
        username = user.get("username", "unknown") if user else "unknown"
        
        # Initialize NSFW detection variables
        is_nsfw = False
        error_message = "NSFW content detected. Please modify your message."
        
        # Use the appropriate detection method
        if self.guardrails_mode == "guardrails":
            try:
                # Validate the message using Guardrails
                validation_result = self.nsfw_validator.guard(user_message)
                
                # Handle different possible return types from guard
                if hasattr(validation_result, 'validated'):
                    is_nsfw = not validation_result.validated
                    if is_nsfw and hasattr(validation_result, 'validation_response'):
                        error_message = f"NSFW content detected: {validation_result.validation_response}"
                
            except Exception as e:
                # Log the error but let the message through
                print(f"Error during Guardrails NSFW validation: {str(e)}")
                # Fall back to basic detection if available
                if self.valves.use_fallback_detection and self.nsfw_pattern:
                    is_nsfw, error_message = self._check_content_fallback(user_message)
        
        elif self.guardrails_mode == "fallback" and self.valves.use_fallback_detection:
            # Use fallback detection
            is_nsfw, error_message = self._check_content_fallback(user_message)
        
        else:
            # No detection available
            print(f"Message from {username} (NSFW detection disabled): {user_message[:30]}...")
            return body
        
        # If NSFW content was detected and blocking is enabled, raise an exception
        if is_nsfw and self.valves.block_nsfw_content:
            print(f"NSFW content detected in message from {username}")
            raise Exception(error_message)
        elif is_nsfw:
            # If blocking is disabled, just log the detection
            print(f"NSFW content detected in message from {username} (not blocked)")
        
        # Content is safe or we're not blocking - return the original body
        return body