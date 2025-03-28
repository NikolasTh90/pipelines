import os
import sys
import time
import traceback
from typing import Optional, List
from pydantic import BaseModel

# Import common utilities from pipelines
from utils.pipelines.logging import get_logger
from utils.pipelines.device import set_device, clear_gpu_memory, print_environment_info
from utils.pipelines.concurrency import run_tasks_concurrently

# Configure logger using utility function
logger = get_logger("RestrictToTopic_Filter")

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline IDs (models) that this filter will be connected to
        pipelines: List[str] = ["*"]
        
        # Priority of the filter
        priority: int = 0

        # Configuration for topic restriction
        valid_topics: List[str] = ["sports"]
        invalid_topics: List[str] = []  # Optionally set topics that should be disallowed
        disable_classifier: bool = True
        disable_llm: bool = False
        model_threshold: float = 0.5
        on_fail: str = "exception"  # Options: exception, fix, filter, refrain, noop, etc.
        
        # Miscellaneous settings
        log_level: str = "info"
        enforce_topic: bool = True   # If True, enforce topic validation (otherwise bypass)
        use_gpu: bool = False        # Default to CPU to avoid CUDA memory issues

    def __init__(self):
        # This filter uses Guardrails’ RestrictToTopic validator to check if text is related to a topic
        self.type = "filter"
        self.name = "RestrictToTopic Filter"

        # Initialize configuration
        self.valves = self.Valves()
        
        # For topic-based validation, keep a reference to the default allowed topic(s)
        self.valid_topics = self.valves.valid_topics
        self.invalid_topics = self.valves.invalid_topics

        # Set device based on configuration
        set_device(self.valves.use_gpu)
        
        # Print environment details via common utility
        print_environment_info(logger)
        
        # Set log level based on configuration
        logger.setLevel(self.valves.log_level.upper())
        
        logger.info("Initializing RestrictToTopic Filter...")
        self.guardrails_mode = self._setup_guardrails()

    def _setup_guardrails(self):
        """
        Sets up Guardrails RestrictToTopic validator.
        Returns "guardrails" if successfully configured; otherwise, "disabled".
        """
        setup_start_time = time.time()
        logger.info("Setting up Guardrails RestrictToTopic validator...")
        
        # First, try to import the basic Guardrails modules
        try:
            import guardrails
            from guardrails import Guard
            from guardrails import hub
            logger.info("Guardrails package and hub module found.")
        except ImportError as e:
            logger.error(f"Guardrails components not available: {e}")
            logger.error("RestrictToTopic validator will be disabled.")
            return "disabled"
        
        # Try to import RestrictToTopic from possible locations
        correct_imports = [
            "from guardrails.hub import RestrictToTopic",
            "from guardrails_hub_restricttotopic import RestrictToTopic",
            "from guardrails_grhub_restricttotopic import RestrictToTopic"
        ]
        restrict_module_found = False
        for imp in correct_imports:
            try:
                logger.info(f"Trying import: {imp}")
                exec(imp, globals())
                logger.info("RestrictToTopic successfully imported!")
                restrict_module_found = True
                break
            except ImportError as e:
                logger.warning(f"Import failed: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error during import: {e}")
        if not restrict_module_found:
            logger.error("Could not import RestrictToTopic from any known location.")
            return "disabled"

        # Try to instantiate the validator and set up a Guard
        try:
            logger.info(f"Instantiating RestrictToTopic with on_fail={self.valves.on_fail} ...")
            # Instantiate the validator with the given config
            # (Assuming the validator accepts these parameters)
            validator = RestrictToTopic(
                valid_topics=self.valves.valid_topics,
                invalid_topics=self.valves.invalid_topics,
                disable_classifier=self.valves.disable_classifier,
                disable_llm=self.valves.disable_llm,
                model_threshold=self.valves.model_threshold,
                on_fail=self.valves.on_fail
            )
            logger.info(f"RestrictToTopic validator created: {validator}")
            
            # Create the Guard that wraps this validator
            from guardrails import Guard
            self.guard = Guard().use(
                RestrictToTopic,
                valid_topics=self.valves.valid_topics,
                invalid_topics=self.valves.invalid_topics,
                disable_classifier=self.valves.disable_classifier,
                disable_llm=self.valves.disable_llm,
                model_threshold=self.valves.model_threshold,
                on_fail=self.valves.on_fail
            )
            logger.info("Guard successfully created using RestrictToTopic validator")
            
            # Optionally test the validator with a sample text
            try:
                logger.info("Testing validator on a sample text...")
                sample_text = "The football match last night was thrilling and full of excitement."
                result = self.guard.validate(sample_text)
                logger.info(f"Validation test outcome: validation_passed={result.validation_passed}")
            except Exception as test_err:
                logger.warning(f"Validation test encountered an error: {test_err}")
            setup_end_time = time.time()
            logger.info(f"RestrictToTopic setup completed in {setup_end_time - setup_start_time:.2f} seconds (guardrails mode)")
            return "guardrails"
        except Exception as e:
            logger.error(f"Error instantiating RestrictToTopic or creating Guard: {e}")
            logger.error("Full traceback:")
            traceback.print_exc()
            return "disabled"

    async def on_startup(self):
        logger.info("RestrictToTopic Filter starting up")
        if self.guardrails_mode == "guardrails":
            logger.info("Running with Guardrails RestrictToTopic validator")
            logger.info(f"Valid topics: {self.valves.valid_topics}")
            logger.info(f"Invalid topics: {self.valves.invalid_topics}")
            logger.info(f"Model threshold: {self.valves.model_threshold}")
            logger.info(f"on_fail policy: {self.valves.on_fail}")
            logger.info(f"Using {'GPU' if self.valves.use_gpu else 'CPU'} for validation")
        else:
            logger.error("RestrictToTopic validator is disabled. Filter will pass text through unmodified.")
    
    async def on_shutdown(self):
        logger.info("RestrictToTopic Filter shutting down")

    async def on_valves_updated(self):
        logger.info("Valves for RestrictToTopic filter updated")
        logger.info(f"Valid topics: {self.valves.valid_topics}")
        logger.info(f"Invalid topics: {self.valves.invalid_topics}")
        logger.info(f"Model threshold: {self.valves.model_threshold}")
        logger.info(f"on_fail policy: {self.valves.on_fail}")
        set_device(self.valves.use_gpu)
        logger.setLevel(self.valves.log_level.upper())
        # If using Guardrails, recreate the Guard with updated settings.
        if self.guardrails_mode == "guardrails":
            try:
                from guardrails import Guard
                self.guard = Guard().use(
                    RestrictToTopic,
                    valid_topics=self.valves.valid_topics,
                    invalid_topics=self.valves.invalid_topics,
                    disable_classifier=self.valves.disable_classifier,
                    disable_llm=self.valves.disable_llm,
                    model_threshold=self.valves.model_threshold,
                    on_fail=self.valves.on_fail
                )
                logger.info("Guard updated with new RestrictToTopic parameters")
            except Exception as e:
                logger.error(f"Error updating Guard: {e}")
                traceback.print_exc()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Inlet passes the message through unchanged.
        """
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process outgoing text:
         1. Validate the text using RestrictToTopic.
         2. If validation fails and enforce_topic is True, modify or block the text.
         3. Clear GPU memory afterwards.
        """
        if body.get("_blocked"):
            if body.get("_raise_block"):
                raise Exception(body.get("_blocked_reason", "Content blocked"))
            else:
                return body
        
        outlet_start = time.time()
        # If Guardrails not available, pass through
        if self.guardrails_mode != "guardrails":
            logger.warning("RestrictToTopic validation disabled - passing content unmodified")
            return body
        
        # Extract text to validate; expect it to be in the last message's content field.
        text = None
        if "messages" in body and body["messages"]:
            text = body["messages"][-1].get("content")
        if not text:
            logger.warning("No text found to validate")
            return body

        logger.info(f"Validating topic for text: {text[:50]}...")
        try:
            # Validate the text using the Guard
            result = self.guard.validate(text)
            if result.validation_passed:
                logger.info("Text passed topic validation; returning unmodified")
                clear_gpu_memory()
                return body
            else:
                logger.info("Text failed topic validation")
                # Depending on on_fail mode, either modify or block content.
                if self.valves.enforce_topic:
                    error_message = "Text does not match the allowed topics."
                    # In this example, we raise an Exception (on_fail "exception")
                    body["_blocked"] = True
                    body["_blocked_reason"] = error_message
                    body["_blocked_by"] = "restricttotopic_filter"
                    body["_raise_block"] = True
                    raise Exception(error_message)
                else:
                    # If not enforcing, simply mark the message.
                    logger.warning("Text marked as invalid but not blocked due to configuration")
                    body["_blocked"] = True
                    body["_blocked_reason"] = "Text topic validation failed"
        except Exception as e:
            logger.error(f"Error during topic validation: {e}")
            traceback.print_exc()
        finally:
            clear_gpu_memory()

        outlet_end = time.time()
        logger.info(f"Outlet processing completed in {outlet_end - outlet_start:.2f} seconds")
        return body
