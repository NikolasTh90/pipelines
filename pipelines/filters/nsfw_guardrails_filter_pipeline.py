import os
import sys
import subprocess
import importlib.util
import time
import traceback
from typing import Optional, List
from pydantic import BaseModel
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NSFW_Filter")

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
        validation_method: str = "sentence"  # 'sentence' or 'full'
        on_fail: str = "exception"  # 'exception', 'noop', 'filter', etc.
        use_gpu: bool = True  # Whether to use GPU for NSFW detection
        
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
        
        # Initialize fallback attributes to ensure they always exist
        self.nsfw_pattern = None
        self.has_alt_detector = False
        
        # Set device for GPU/CPU
        if self.valves.use_gpu:
            os.environ["DEVICE"] = "cuda"
            logger.info("Set NSFW validator to use GPU (CUDA)")
        else:
            os.environ["DEVICE"] = "cpu" 
            logger.info("Set NSFW validator to use CPU")
        
        # Print diagnostic information
        self._print_environment_info()
        
        # Set logging level based on configuration
        if hasattr(logging, self.valves.log_level.upper()):
            logger.setLevel(getattr(logging, self.valves.log_level.upper()))
        
        # Try to import Guardrails components
        logger.info("Initializing Guardrails NSFW Filter...")
        self.guardrails_mode = self._setup_guardrails()
        
        # Setup enhanced fallback if guardrails not available or as additional layer
        if self.valves.use_fallback_detection:
            self._setup_enhanced_fallback()
            logger.info("Enhanced fallback detection set up successfully")

    def _print_environment_info(self):
        """Print diagnostic information about the environment"""
        logger.info("=== Environment Information ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        logger.info(f"DEVICE: {os.environ.get('DEVICE', 'Not set')}")
        logger.info(f"sys.path: {sys.path}")
        
        # Check GPU availability if using GPU
        if self.valves.use_gpu:
            try:
                import torch
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            except ImportError:
                logger.warning("Torch not available - cannot check CUDA status")
        
        # Check if guardrails is in the path
        guardrails_locations = [p for p in sys.path if 'guardrails' in p]
        if guardrails_locations:
            logger.info(f"Potential guardrails locations in path: {guardrails_locations}")
        
        # List available modules
        try:
            import pkgutil
            guardrails_specs = [p for p in pkgutil.iter_modules() if 'guardrails' in p.name]
            logger.info(f"Available guardrails modules: {guardrails_specs}")
        except Exception as e:
            logger.warning(f"Error listing modules: {e}")
        
        logger.info("===============================")

    def _setup_guardrails(self):
        """Set up Guardrails and return the mode (guardrails, fallback, or disabled)"""
        logger.info("Setting up Guardrails NSFW validator...")
        
        # First check if guardrails is installed
        try:
            import guardrails
            logger.info(f"Guardrails package found. Checking version...")
            try:
                logger.info(f"Guardrails version: {guardrails.__version__}")
            except AttributeError:
                logger.warning("Could not determine Guardrails version")
        except ImportError as e:
            logger.warning(f"Guardrails package not found: {e}")
            # Attempt installation if configured
            if self.valves.attempt_auto_install:
                logger.info("Attempting to install Guardrails...")
                if self._install_guardrails():
                    try:
                        import guardrails
                        logger.info("Guardrails successfully installed and imported")
                    except ImportError as e2:
                        logger.error(f"Could not import Guardrails after installation: {e2}")
                        return "fallback"
                else:
                    logger.error("Failed to install Guardrails")
                    return "fallback"
            else:
                return "fallback"
        
        # Now check for Guard class
        try:
            from guardrails import Guard
            logger.info("Guardrails Guard class successfully imported")
        except ImportError as e:
            logger.error(f"Could not import Guard from guardrails: {e}")
            return "fallback"
        
        # Now check for hub module and contents
        try:
            from guardrails import hub
            logger.info("Guardrails hub module found")
            logger.info(f"Hub contents: {dir(hub)}")
        except ImportError as e:
            logger.error(f"Could not import hub from guardrails: {e}")
            return "fallback"
            
        # Try to import NSFWText specifically - try multiple possible locations
        nsfw_module_found = False
        for import_path in [
            "from guardrails.hub import NSFWText",
            "from guardrails_hub_nsfw_text import NSFWText",
            "from guardrails_grhub_nsfw_text import NSFWText"
        ]:
            try:
                logger.info(f"Trying import: {import_path}")
                exec(import_path, globals())
                logger.info("NSFWText successfully imported!")
                nsfw_module_found = True
                break
            except ImportError as e:
                logger.warning(f"Import failed: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error during import: {e}")
        
        if not nsfw_module_found:
            logger.error("Could not import NSFWText from any known location")
            # Last attempt - try to install from hub directly
            if self.valves.attempt_auto_install:
                logger.info("Attempting to install NSFWText validator from hub...")
                if self._install_nsfw_text():
                    # Try imports again
                    for import_path in [
                        "from guardrails.hub import NSFWText",
                        "from guardrails_hub_nsfw_text import NSFWText",
                        "from guardrails_grhub_nsfw_text import NSFWText"
                    ]:
                        try:
                            logger.info(f"Trying import after installation: {import_path}")
                            exec(import_path, globals())
                            logger.info("NSFWText successfully imported after installation!")
                            nsfw_module_found = True
                            break
                        except ImportError as e:
                            logger.warning(f"Import failed after installation: {e}")
                        except Exception as e:
                            logger.warning(f"Unexpected error during import after installation: {e}")
            
            if not nsfw_module_found:
                logger.error("NSFWText could not be imported after all attempts")
                return "fallback"
        
        # Try to instantiate NSFWText and create Guard
        try:
            # Test instantiation - store the on_fail mode to handle validation results appropriately
            logger.info(f"Attempting to instantiate NSFWText with on_fail={self.valves.on_fail}...")
            
            # Store the original on_fail mode for later reference
            self.on_fail_mode = self.valves.on_fail
            logger.info(f"Using on_fail mode: {self.on_fail_mode}")
            
            validator = NSFWText(
                threshold=self.valves.threshold,
                validation_method=self.valves.validation_method,
                on_fail=self.on_fail_mode
            )
            logger.info(f"NSFWText validator created: {validator}")
            
            # Create Guard
            from guardrails import Guard
            logger.info("Creating Guard with NSFWText validator...")
            self.guard = Guard().use(
                NSFWText,
                threshold=self.valves.threshold,
                validation_method=self.valves.validation_method,
                on_fail=self.on_fail_mode
            )
            logger.info("Guard successfully created with NSFWText")
            
            # Test validation with a simple string
            try:
                logger.info("Testing validator with a simple string...")
                result = self.guard.validate("This is a test string.")
                
                # Log the validation result in a way that works for all on_fail modes
                logger.info(f"Validation outcome: validation_passed={result.validation_passed}")
                logger.info(f"Original text: 'This is a test string.'")
                
                # Check if validated_output exists and is not None before slicing
                if hasattr(result, 'validated_output') and result.validated_output is not None:
                    logger.info(f"Validated output: '{result.validated_output}'")
                else:
                    logger.info("Validated output: None")
                
                # Log if the device is actually being used
                logger.info(f"Device being used: {os.environ.get('DEVICE', 'Not reported')}")
            except Exception as e:
                logger.warning(f"Validation test failed: {e}")
                logger.warning("This may indicate issues with the validator, but we'll continue")
                # Don't return fallback yet, as validation might work for real inputs
            
            return "guardrails"
        except Exception as e:
            logger.error(f"Error setting up NSFWText or Guard: {e}")
            logger.error("Full traceback:")
            traceback.print_exc()
            return "fallback"
                
    def _install_guardrails(self):
        """Attempt to install Guardrails using subprocess"""
        try:
            # First install the guardrails-ai package
            logger.info("Installing guardrails-ai package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "guardrails-ai"])
            
            # Wait a moment for installation to complete
            time.sleep(2)
            logger.info("Guardrails-ai package installation completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Installation failed: {str(e)}")
            logger.info("To install manually, run:")
            logger.info("  pip install guardrails-ai")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during installation: {str(e)}")
            return False

    def _install_nsfw_text(self):
        """Attempt to install NSFWText validator from hub"""
        try:
            # Install system dependencies (in case they're needed)
            logger.info("Installing system dependencies for NSFWText...")
            try:
                subprocess.check_call([
                    "apt-get", "update", "-y"
                ])
                subprocess.check_call([
                    "apt-get", "install", "-y",
                    "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
                    "libxext6", "libxrender-dev"
                ])
                logger.info("System dependencies installed successfully")
            except Exception as e:
                logger.warning(f"Could not install system dependencies: {e}")
                logger.warning("Continuing with hub installation anyway...")
            
            # Install NSFWText from hub
            logger.info("Installing NSFWText validator from Guardrails hub...")
            subprocess.check_call(["guardrails", "hub", "install", "hub://guardrails/nsfw_text"])
            time.sleep(2)

            # Try direct pip installation as fallback
            try:
                logger.info("Also trying direct pip installation of nsfw_text...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "guardrails-hub-nsfw-text"
                ])
            except Exception as e:
                logger.warning(f"Direct pip installation failed: {e}")
                logger.warning("Continuing with hub installation only")

            logger.info("NSFWText installation completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"NSFWText installation failed: {str(e)}")
            logger.error("To install manually, run:")
            logger.error("  guardrails hub install hub://guardrails/nsfw_text")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during NSFWText installation: {str(e)}")
            return False

    def _setup_enhanced_fallback(self):
        """Set up enhanced fallback NSFW detection"""
        logger.info("Setting up enhanced fallback NSFW detection")
        
        # A more comprehensive list of NSFW terms
        nsfw_terms = [
            # Common explicit terms
            'explicit', 'nsfw', 'xxx', 'porn', 'pornography', 'sexual', 'obscene',
            # Sexual content
            'intercourse', 'masturbat', 'orgasm', 'erotic', 'erotica', 'foreplay',
            # Profanity (common ones)
            'fuck', 'shit', 'ass', 'damn', 'bitch', 'cunt', 'dick', 'cock', 'pussy',
            # Violence
            'gore', 'mutilation', 'torture', 'rape', 'abuse'
        ]
        
        # Create a pattern that handles word boundaries
        pattern = r'\b(' + '|'.join(nsfw_terms) + r')\b'
        self.nsfw_pattern = re.compile(pattern, re.IGNORECASE)
        logger.info(f"Enhanced fallback regex pattern created with {len(nsfw_terms)} terms")
        
        # Try to set up alternative detector if available
        try:
            from profanity_check import predict_prob
            self.alt_detector = predict_prob
            logger.info("Successfully set up alternative profanity detector")
            self.has_alt_detector = True
        except ImportError:
            logger.info("Profanity-check not available, not using alternative detector")
            self.has_alt_detector = False
            # Try to install it
            try:
                logger.info("Attempting to install profanity-check as alternative detector...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "profanity-check"
                ])
                from profanity_check import predict_prob
                self.alt_detector = predict_prob
                self.has_alt_detector = True
                logger.info("Successfully installed and set up alternative profanity detector")
            except Exception as e:
                logger.warning(f"Could not install alternative detector: {e}")

    async def on_startup(self):
        # This function is called when the server is started
        logger.info(f"Guardrails NSFW Filter starting up")
        
        if self.guardrails_mode == "guardrails":
            logger.info(f"Running with Guardrails NSFWText validator")
            logger.info(f"NSFW detection threshold: {self.valves.threshold}")
            logger.info(f"Validation method: {self.valves.validation_method}")
            logger.info(f"On fail action: {self.valves.on_fail}")
            logger.info(f"Using {'GPU' if self.valves.use_gpu else 'CPU'} for detection")
        elif self.guardrails_mode == "fallback" and self.valves.use_fallback_detection:
            logger.info("Running in fallback mode with enhanced pattern matching")
            if hasattr(self, 'has_alt_detector') and self.has_alt_detector:
                logger.info("Also using alternative profanity detector")
            logger.info("For better detection, install the full Guardrails NSFWText validator:")
            logger.info("  pip install guardrails-ai")
            logger.info("  guardrails hub install hub://guardrails/nsfw_text")
        else:
            logger.info("NSFW detection disabled")
            
        logger.info(f"Block NSFW content: {self.valves.block_nsfw_content}")

    async def on_shutdown(self):
        # This function is called when the server is stopped
        logger.info("Guardrails NSFW Filter shutting down")

    async def on_valves_updated(self):
        # This function is called when the valves are updated
        logger.info(f"Valves updated: NSFW threshold = {self.valves.threshold}")
        logger.info(f"Block NSFW content: {self.valves.block_nsfw_content}")
        logger.info(f"Validation method: {self.valves.validation_method}")
        logger.info(f"On fail action: {self.valves.on_fail}")
        logger.info(f"Using GPU: {self.valves.use_gpu}")
        
        # If on_fail mode changed, store it
        if hasattr(self, 'on_fail_mode') and self.on_fail_mode != self.valves.on_fail:
            self.on_fail_mode = self.valves.on_fail
            logger.info(f"Updated on_fail mode to: {self.on_fail_mode}")
        
        # Update GPU/CPU setting if it changed
        if self.valves.use_gpu:
            os.environ["DEVICE"] = "cuda"
            logger.info("Set NSFW validator to use GPU (CUDA)")
        else:
            os.environ["DEVICE"] = "cpu" 
            logger.info("Set NSFW validator to use CPU")
        
        # Update the logging level if it changed
        if hasattr(logging, self.valves.log_level.upper()):
            logger.setLevel(getattr(logging, self.valves.log_level.upper()))
            logger.info(f"Updated log level to {self.valves.log_level.upper()}")
        
        # Update the validator if we're using Guardrails
        if self.guardrails_mode == "guardrails":
            try:
                # We use the global NSFWText that was successfully imported earlier
                from guardrails import Guard
                
                # Recreate the Guard with updated parameters
                self.guard = Guard().use(
                    NSFWText,
                    threshold=self.valves.threshold,
                    validation_method=self.valves.validation_method,
                    on_fail=self.valves.on_fail
                )
                logger.info(f"Updated Guardrails Guard with new parameters")
            except Exception as e:
                logger.error(f"Error updating Guardrails Guard: {str(e)}")
                traceback.print_exc()
                logger.warning("Falling back to regex pattern detection")
                self.guardrails_mode = "fallback"
                self._setup_enhanced_fallback()
        
        # Always ensure fallback is set up, whether we're in fallback mode or not
        if self.valves.use_fallback_detection:
            self._setup_enhanced_fallback()

    def _check_content_fallback(self, text):
        """Fallback NSFW detection using regex patterns and alternative detector"""
        results = []
        
        # Check with regex pattern
        if self.nsfw_pattern:
            logger.debug(f"Running regex NSFW detection on text: {text[:30]}...")
            if self.nsfw_pattern.search(text):
                results.append(("regex", True, "Potentially NSFW content detected based on keyword matching"))
            else:
                results.append(("regex", False, ""))
        else:
            logger.warning("Regex pattern not available for fallback detection")
        
        # Check with alternative detector if available
        if hasattr(self, 'has_alt_detector') and self.has_alt_detector:
            try:
                logger.debug("Running alternative profanity detector...")
                nsfw_score = self.alt_detector([text])[0]
                is_nsfw = nsfw_score > self.valves.threshold
                if is_nsfw:
                    results.append(("alt", True, f"NSFW content detected with confidence score: {nsfw_score:.2f}"))
                else:
                    results.append(("alt", False, ""))
            except Exception as e:
                logger.warning(f"Error in alternative detector: {e}")
        
        # If any detector found NSFW content, report it
        for detector, is_nsfw, message in results:
            if is_nsfw:
                logger.info(f"NSFW content detected by {detector} detector")
                return True, message
        
        # If we got here, no detection method reported NSFW content
        if not results:
            logger.warning("No fallback detection methods were available")
            return False, "No detection methods available"
            
        logger.debug("No NSFW content detected by any fallback mechanism")
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
        # Safely check if content is already blocked
        if body.get("_blocked"):
            if body.get("_raise_block"):
                raise Exception(body.get("_blocked_reason", "Content blocked"))
            else:
                return body
        if "messages" not in body or not body["messages"]:
            logger.debug("No messages found in request body")
            return body
            
        user_message = body["messages"][-1]["content"]
        username = user.get("username", "unknown") if user else "unknown"
        
        logger.info(f"Processing message from {username}: {user_message[:30]}...")
        
        # Initialize NSFW detection variables
        is_nsfw = False
        error_message = "NSFW content detected. Please modify your message."
        
        # Use the appropriate detection method
        if self.guardrails_mode == "guardrails":
            logger.debug("Using Guardrails for NSFW detection")
            start_time = time.time()
            
            # Different handling based on on_fail mode
            if self.on_fail_mode == "exception":
                # For exception mode, we catch exceptions
                try:
                    validation_result = self.guard.validate(user_message)
                    end_time = time.time()
                    logger.debug(f"Message passed NSFW validation in {end_time - start_time:.2f} seconds")
                    is_nsfw = False
                except Exception as validation_error:
                    end_time = time.time()
                    logger.info(f"NSFW content detected in {end_time - start_time:.2f} seconds: {str(validation_error)}")
                    is_nsfw = True
                    error_message = str(validation_error)
            else:
                # For all other modes, validation doesn't raise exceptions, so we check the result
                try:
                    validation_result = self.guard.validate(user_message)
                    end_time = time.time()
                    
                    # Check if validation failed
                    if not validation_result.validation_passed:
                        logger.info(f"NSFW content detected in {end_time - start_time:.2f} seconds")
                        logger.info(f"Original text: '{user_message[:50]}...'")
                        
                        # Safely check and log validated_output
                        if hasattr(validation_result, 'validated_output') and validation_result.validated_output is not None:
                            validated_output = validation_result.validated_output
                            logger.info(f"Validated output: '{validated_output[:50]}...'")
                        else:
                            logger.info("Validated output: None")
                        
                        # Extract validation summaries for detailed error messages
                        if hasattr(validation_result, 'validation_summaries') and validation_result.validation_summaries:
                            for summary in validation_result.validation_summaries:
                                if hasattr(summary, 'message'):
                                    error_message = summary.message
                                    break
                        
                        is_nsfw = True
                        
                        # Special handling for different on_fail modes
                        if self.on_fail_mode == "refrain":
                            logger.info("Using 'refrain' mode - replacing with safe content")
                            # In refrain mode, we would normally just use validation_result.validated_output
                            # But since we want to block it, we treat it as NSFW
                        elif self.on_fail_mode == "filter":
                            logger.info("Using 'filter' mode - content would be filtered")
                            # Same as above - for our filter we still treat it as NSFW
                        elif self.on_fail_mode == "noop":
                            logger.info("Using 'noop' mode - content passed but marked as NSFW")
                    else:
                        logger.debug(f"Message passed NSFW validation in {end_time - start_time:.2f} seconds")
                        is_nsfw = False
                except Exception as e:
                    # Unexpected error during validation
                    end_time = time.time()
                    logger.error(f"Unexpected error during Guardrails NSFW validation: {str(e)}")
                    logger.error("Full traceback:")
                    traceback.print_exc()
                    
                    # Fall back to alternative detection if available
                    if self.valves.use_fallback_detection:
                        logger.info("Falling back to alternative detection methods")
                        # Ensure fallback is set up
                        if self.nsfw_pattern is None:
                            self._setup_enhanced_fallback()
                        is_nsfw, error_message = self._check_content_fallback(user_message)
        
        elif self.guardrails_mode == "fallback" and self.valves.use_fallback_detection:
            # Use fallback detection
            logger.debug("Using fallback NSFW detection methods")
            start_time = time.time()
            # Ensure fallback is set up
            if self.nsfw_pattern is None:
                self._setup_enhanced_fallback()
            is_nsfw, error_message = self._check_content_fallback(user_message)
            end_time = time.time()
            logger.debug(f"Fallback detection completed in {end_time - start_time:.2f} seconds")
        
        else:
            # No detection available
            logger.info(f"Message from {username} processed (NSFW detection disabled)")
            return body
        
        # If NSFW content was detected and blocking is enabled, raise an exception
        if is_nsfw and self.valves.block_nsfw_content:
            logger.warning(f"NSFW content detected in message from {username} - BLOCKED")
            body["_blocked"] = True
            body["_blocked_reason"] = error_message
            body["_blocked_by"] = "nsfw_filter"
            body["_raise_block"] = True
            raise Exception(error_message)
        elif is_nsfw:
            # If blocking is disabled, just log the detection
            logger.warning(f"NSFW content detected in message from {username} (not blocked)")
            body["_blocked"] = True
            body["_blocked_reason"] = error_message
            body["_blocked_by"] = "nsfw_filter"
        else:
            logger.info(f"Message from {username} passed NSFW check")
        
        # Content is safe or we're not blocking - return the original body
        return body